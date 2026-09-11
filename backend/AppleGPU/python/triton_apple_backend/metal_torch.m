// metal_torch - Triton Apple backend Metal runtime. Links against the user's
// libtorch for getMTLBufferStorage() (zero-copy MPS tensor dispatch).
// Requires PyTorch 2.0+ with MPS.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// getMTLBufferStorage mirrors PyTorch's ATen/native/mps/OperationUtils.h.
#include <ATen/Tensor.h>
#include <ATen/mps/MPSStream.h>
#include <torch/csrc/autograd/python_variable.h>

static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase &t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

// Use PyTorch's MPS device - same device that owns the tensor buffers.
static id<MTLDevice> get_device(void) {
  return at::mps::getCurrentMPSStream()->device();
}

// ── MetalKernel - callable PSO wrapper ───────────────────────────────────

typedef struct {
  PyObject_HEAD id<MTLComputePipelineState> pso;
  NSUInteger maxThreads;
} MetalKernelObject;

static void MetalKernel_dealloc(MetalKernelObject *self) {
  self->pso = nil;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// One kernel argument, read out of Python before the dispatch block: Python
// API calls are not safe inside dispatch_sync.
// The launcher passes MPS tensors for pointers and one `bytes` blob holding
// every scalar, packed per agpu/emit/KernelAbi.h. Bare ints and floats never
// reach here.
struct ArgInfo {
  enum Kind { TENSOR, BYTES } kind;
  id<MTLBuffer> buf;
  NSUInteger offset;
  const void *bytesPtr;
  Py_ssize_t bytesLen;
};

struct LaunchGeometry {
  long tx, ty, tz;
  long gx, gy, gz;
  long tgmem;
};

// False with the Python error set.
static bool readLaunchGeometry(PyObject *kwargs, LaunchGeometry *out) {
  PyObject *threads_obj = NULL, *group_obj = NULL, *tgmem_obj = NULL;
  if (kwargs) {
    threads_obj = PyDict_GetItemString(kwargs, "threads");
    group_obj = PyDict_GetItemString(kwargs, "group_size");
    tgmem_obj = PyDict_GetItemString(kwargs, "threadgroup_mem");
  }
  if (!threads_obj || !group_obj) {
    PyErr_SetString(PyExc_ValueError, "threads and group_size required");
    return false;
  }

  // Dynamic threadgroup memory: when nonzero the kernel declares a trailing
  // addrspace(3) param; bind its byte length at TG location-index 0 (a separate
  // index space from device setBuffer, so no clash with the buffer args).
  out->tgmem = tgmem_obj ? PyLong_AsLong(tgmem_obj) : 0;

  out->tx = PyLong_AsLong(PyList_GetItem(threads_obj, 0));
  out->ty = PyLong_AsLong(PyList_GetItem(threads_obj, 1));
  out->tz = PyLong_AsLong(PyList_GetItem(threads_obj, 2));
  out->gx = PyLong_AsLong(PyList_GetItem(group_obj, 0));
  out->gy = PyLong_AsLong(PyList_GetItem(group_obj, 1));
  out->gz = PyLong_AsLong(PyList_GetItem(group_obj, 2));
  return true;
}

// False with the Python error set.
static bool packArguments(PyObject *args, std::vector<ArgInfo> *out) {
  const Py_ssize_t nargs = PyTuple_Size(args);
  out->resize(nargs);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject *arg = PyTuple_GetItem(args, i);
    ArgInfo &info = (*out)[i];
    if (THPVariable_Check(arg)) {
      at::Tensor t = THPVariable_Unpack(arg);
      // Guard before t.device(): a storage-less tensor (ABI-skewed libtorch)
      // makes t.device() abort with an uncatchable c10::Error.
      if (!t.defined() || !t.has_storage()) {
        PyErr_Format(PyExc_RuntimeError,
                     "Arg %zd: tensor is undefined or has no storage "
                     "(rebuild metal_torch against the current libtorch?)",
                     i);
        return false;
      }
      if (!t.is_mps()) {
        PyErr_Format(PyExc_RuntimeError,
                     "Arg %zd: tensor must be on MPS device, got %s", i,
                     t.device().str().c_str());
        return false;
      }
      info.kind = ArgInfo::TENSOR;
      info.buf = getMTLBufferStorage(t);
      info.offset = t.storage_offset() * t.element_size();
    } else if (PyBytes_Check(arg)) {
      // Packed scalar blob, bound inline via setBytes. The args tuple keeps
      // the object alive across the dispatch_sync below.
      info.kind = ArgInfo::BYTES;
      info.bytesPtr = PyBytes_AS_STRING(arg);
      info.bytesLen = PyBytes_GET_SIZE(arg);
    } else {
      PyErr_Format(PyExc_TypeError, "Arg %zd: expected an MPS tensor or the "
                                    "packed scalar bytes", i);
      return false;
    }
  }
  return true;
}

static void bindArguments(id<MTLComputeCommandEncoder> enc,
                          const std::vector<ArgInfo> &argInfos) {
  for (size_t i = 0; i < argInfos.size(); i++) {
    const ArgInfo &info = argInfos[i];
    switch (info.kind) {
    case ArgInfo::TENSOR:
      [enc setBuffer:info.buf offset:info.offset atIndex:i];
      break;
    case ArgInfo::BYTES:
      [enc setBytes:info.bytesPtr length:(NSUInteger)info.bytesLen atIndex:i];
      break;
    }
  }
}

static void encodeDispatch(MetalKernelObject *self, const LaunchGeometry &geom,
                           const std::vector<ArgInfo> &argInfos) {
  // Dispatch on stream->queue() (serial) to serialize with other MPS ops.
  // Don't call endKernelCoalescing(): reusing torch's cached encoder
  // coalesces back-to-back dispatches and RAW deps are still honored.
  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

        [enc setComputePipelineState:self->pso];

        if (geom.tgmem > 0)
          [enc setThreadgroupMemoryLength:geom.tgmem atIndex:0];

        bindArguments(enc, argInfos);

        MTLSize threadgroups = MTLSizeMake(geom.tx / geom.gx, geom.ty / geom.gy,
                                           geom.tz / geom.gz);
        MTLSize threadsPerGroup = MTLSizeMake(geom.gx, geom.gy, geom.gz);
        [enc dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerGroup];
      }
    });
  }
}

static PyObject *MetalKernel_call(MetalKernelObject *self, PyObject *args,
                                  PyObject *kwargs) {
  LaunchGeometry geom;
  if (!readLaunchGeometry(kwargs, &geom))
    return NULL;

  std::vector<ArgInfo> argInfos;
  if (!packArguments(args, &argInfos))
    return NULL;

  encodeDispatch(self, geom, argInfos);
  Py_RETURN_NONE;
}

static PyObject *MetalKernel_get_max_threads(MetalKernelObject *self,
                                             void *closure) {
  return PyLong_FromUnsignedLongLong(self->maxThreads);
}

static PyGetSetDef MetalKernel_getset[] = {{"max_total_threads_per_threadgroup",
                                            (getter)MetalKernel_get_max_threads,
                                            NULL, NULL, NULL},
                                           {NULL}};

static PyTypeObject MetalKernelType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "metal_torch.MetalKernel",
    .tp_basicsize = sizeof(MetalKernelObject),
    .tp_dealloc = (destructor)MetalKernel_dealloc,
    .tp_call = (ternaryfunc)MetalKernel_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = MetalKernel_getset,
};

// ── MetalLibrary - metallib container ────────────────────────────────────

typedef struct {
  PyObject_HEAD id<MTLLibrary> library;
} MetalLibraryObject;

static void MetalLibrary_dealloc(MetalLibraryObject *self) {
  self->library = nil;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MetalLibrary_get_function(MetalLibraryObject *self,
                                           PyObject *args) {
  const char *name;
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;

  NSString *fnName = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [self->library newFunctionWithName:fnName];
  if (!fn) {
    PyErr_Format(PyExc_KeyError, "Function '%s' not found", name);
    return NULL;
  }

  NSError *error = nil;
  id<MTLComputePipelineState> pso =
      [get_device() newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) {
    // The PSO compiler often leaves localizedDescription useless and puts the
    // real diagnostic in userInfo or the underlying error.
    NSMutableString *full = [NSMutableString string];
    [full appendFormat:@"%@", [error localizedDescription]];
    NSDictionary *info = [error userInfo];
    if (info && [info count])
      [full appendFormat:@" | userInfo=%@", info];
    NSError *under = [[error userInfo] objectForKey:NSUnderlyingErrorKey];
    if (under)
      [full appendFormat:@" | underlying=%@ (%@)", [under localizedDescription],
                         [under userInfo]];
    PyErr_Format(PyExc_RuntimeError, "PSO creation failed: %s",
                 [full UTF8String]);
    return NULL;
  }

  MetalKernelObject *kernel = PyObject_New(MetalKernelObject, &MetalKernelType);
  kernel->pso = pso;
  kernel->maxThreads = [pso maxTotalThreadsPerThreadgroup];
  return (PyObject *)kernel;
}

static PyMethodDef MetalLibrary_methods[] = {
    {"get_function", (PyCFunction)MetalLibrary_get_function, METH_VARARGS,
     NULL},
    {NULL}};

static PyTypeObject MetalLibraryType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "metal_torch.MetalLibrary",
    .tp_basicsize = sizeof(MetalLibraryObject),
    .tp_dealloc = (destructor)MetalLibrary_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = MetalLibrary_methods,
};

// ── Module functions ─────────────────────────────────────────────────────

static PyObject *py_load_metallib(PyObject *self, PyObject *args) {
  Py_buffer buf;
  if (!PyArg_ParseTuple(args, "y*", &buf))
    return NULL;

  @autoreleasepool {
    dispatch_data_t dd = dispatch_data_create(buf.buf, buf.len, nil,
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    PyBuffer_Release(&buf);

    NSError *error = nil;
    id<MTLLibrary> lib = [get_device() newLibraryWithData:dd error:&error];
    if (!lib) {
      PyErr_Format(PyExc_RuntimeError, "Failed to load metallib: %s",
                   [[error localizedDescription] UTF8String]);
      return NULL;
    }

    MetalLibraryObject *obj =
        PyObject_New(MetalLibraryObject, &MetalLibraryType);
    obj->library = lib;
    return (PyObject *)obj;
  }
}

static PyObject *py_is_available(PyObject *self, PyObject *Py_UNUSED(args)) {
  return PyBool_FromLong(MTLCreateSystemDefaultDevice() != nil);
}

static PyMethodDef module_methods[] = {
    {"load_metallib", py_load_metallib, METH_VARARGS, NULL},
    {"is_available", py_is_available, METH_NOARGS, NULL},
    {NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "metal_torch",
    "Triton Metal runtime: zero-copy MPS dispatch via libtorch", -1,
    module_methods};

PyMODINIT_FUNC PyInit_metal_torch(void) {
  PyObject *m = PyModule_Create(&module_def);
  if (!m)
    return NULL;
  if (PyType_Ready(&MetalKernelType) < 0)
    return NULL;
  if (PyType_Ready(&MetalLibraryType) < 0)
    return NULL;
  Py_INCREF(&MetalKernelType);
  Py_INCREF(&MetalLibraryType);
  PyModule_AddObject(m, "MetalKernel", (PyObject *)&MetalKernelType);
  PyModule_AddObject(m, "MetalLibrary", (PyObject *)&MetalLibraryType);
  return m;
}
