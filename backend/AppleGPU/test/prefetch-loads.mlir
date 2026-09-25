// RUN: apple_opt.py %s prefetch_loads=2 | %filecheck %s
// RUN: apple_opt.py %s prefetch_loads | %filecheck %s --check-prefix=OFF

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#opA = #ttg.dot_op<{opIdx = 0, parent = #acc}>
#opB = #ttg.dot_op<{opIdx = 1, parent = #acc}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {
  // The first iteration's loads move before the loop under the trip-count
  // predicate, each iteration loads the next one's operands before its dot,
  // and the last dot runs after the loop.
  // CHECK-LABEL: @dot_loop
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
  // CHECK: scf.for
  // CHECK-SAME: tensor<32x32xf32, #blocked>, tensor<32x32xf32, #blocked>
  // CHECK: tt.load
  // CHECK: tt.load
  // CHECK: tt.dot
  // CHECK: scf.yield
  // CHECK: tt.dot
  // OFF-LABEL: @dot_loop
  // OFF-NOT: tt.load
  // OFF: scf.for
  // OFF-NEXT: tt.load
  tt.func @dot_loop(%a: tensor<32x32x!tt.ptr<f32>, #blocked>, %b: tensor<32x32x!tt.ptr<f32>, #blocked>, %step: tensor<32x32xi32, #blocked>, %k: i32) -> tensor<32x32xf32, #acc> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #acc>
    %r:3 = scf.for %i = %c0 to %k step %c1 iter_args(%acc = %zero, %pa = %a, %pb = %b) -> (tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>)  : i32 {
      %x = tt.load %pa : tensor<32x32x!tt.ptr<f32>, #blocked>
      %y = tt.load %pb : tensor<32x32x!tt.ptr<f32>, #blocked>
      %xa = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
      %yb = ttg.convert_layout %y : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
      %d = tt.dot %xa, %yb, %acc : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
      %na = tt.addptr %pa, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %nb = tt.addptr %pb, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %d, %na, %nb : tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    }
    tt.return %r#0 : tensor<32x32xf32, #acc>
  }

  // The next A address depends on this iteration's dot, so its load cannot
  // run ahead and the loop stays as it is.
  // CHECK-LABEL: @address_from_dot
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  tt.func @address_from_dot(%a: tensor<32x32x!tt.ptr<f32>, #blocked>, %b: tensor<32x32x!tt.ptr<f32>, #blocked>, %k: i32) -> tensor<32x32xf32, #acc> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #acc>
    %r:2 = scf.for %i = %c0 to %k step %c1 iter_args(%acc = %zero, %pa = %a) -> (tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>)  : i32 {
      %x = tt.load %pa : tensor<32x32x!tt.ptr<f32>, #blocked>
      %y = tt.load %b : tensor<32x32x!tt.ptr<f32>, #blocked>
      %xa = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
      %yb = ttg.convert_layout %y : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
      %d = tt.dot %xa, %yb, %acc : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
      %off = arith.fptosi %d : tensor<32x32xf32, #acc> to tensor<32x32xi32, #acc>
      %offb = ttg.convert_layout %off : tensor<32x32xi32, #acc> -> tensor<32x32xi32, #blocked>
      %na = tt.addptr %pa, %offb : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %d, %na : tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>
    }
    tt.return %r#0 : tensor<32x32xf32, #acc>
  }

  // The inner K loop runs its loads ahead; the outer loop, which would carry
  // its own across the whole inner loop, does not.
  // CHECK-LABEL: @outer_loop
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
  // CHECK: tt.load %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
  // CHECK: scf.for
  tt.func @outer_loop(%a: tensor<32x32x!tt.ptr<f32>, #blocked>, %b: tensor<32x32x!tt.ptr<f32>, #blocked>, %step: tensor<32x32xi32, #blocked>, %k: i32, %n: i32) -> tensor<32x32xf32, #acc> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #acc>
    %o = scf.for %j = %c0 to %n step %c1 iter_args(%sum = %zero) -> (tensor<32x32xf32, #acc>)  : i32 {
      %q = tt.load %a : tensor<32x32x!tt.ptr<f32>, #blocked>
      %qa = ttg.convert_layout %q : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
      %qb = ttg.convert_layout %q : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
      %s = tt.dot %qa, %qb, %sum : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
      %r:3 = scf.for %i = %c0 to %k step %c1 iter_args(%acc = %s, %pa = %a, %pb = %b) -> (tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>)  : i32 {
        %x = tt.load %pa : tensor<32x32x!tt.ptr<f32>, #blocked>
        %y = tt.load %pb : tensor<32x32x!tt.ptr<f32>, #blocked>
        %xa = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
        %yb = ttg.convert_layout %y : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
        %d = tt.dot %xa, %yb, %acc : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
        %na = tt.addptr %pa, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %nb = tt.addptr %pb, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        scf.yield %d, %na, %nb : tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
      }
      scf.yield %r#0 : tensor<32x32xf32, #acc>
    }
    tt.return %o : tensor<32x32xf32, #acc>
  }

  // A loop's own num_stages beats the pass option, in both directions.
  // CHECK-LABEL: @loop_asks_one
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  // OFF-LABEL: @loop_asks_two
  // OFF: tt.load %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
  // OFF: scf.for
  tt.func @loop_asks_one(%a: tensor<32x32x!tt.ptr<f32>, #blocked>, %b: tensor<32x32x!tt.ptr<f32>, #blocked>, %step: tensor<32x32xi32, #blocked>, %k: i32) -> tensor<32x32xf32, #acc> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #acc>
    %r:3 = scf.for %i = %c0 to %k step %c1 iter_args(%acc = %zero, %pa = %a, %pb = %b) -> (tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>)  : i32 {
      %x = tt.load %pa : tensor<32x32x!tt.ptr<f32>, #blocked>
      %y = tt.load %pb : tensor<32x32x!tt.ptr<f32>, #blocked>
      %xa = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
      %yb = ttg.convert_layout %y : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
      %d = tt.dot %xa, %yb, %acc : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
      %na = tt.addptr %pa, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %nb = tt.addptr %pb, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %d, %na, %nb : tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 1 : i32}
    tt.return %r#0 : tensor<32x32xf32, #acc>
  }
  tt.func @loop_asks_two(%a: tensor<32x32x!tt.ptr<f32>, #blocked>, %b: tensor<32x32x!tt.ptr<f32>, #blocked>, %step: tensor<32x32xi32, #blocked>, %k: i32) -> tensor<32x32xf32, #acc> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #acc>
    %r:3 = scf.for %i = %c0 to %k step %c1 iter_args(%acc = %zero, %pa = %a, %pb = %b) -> (tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>)  : i32 {
      %x = tt.load %pa : tensor<32x32x!tt.ptr<f32>, #blocked>
      %y = tt.load %pb : tensor<32x32x!tt.ptr<f32>, #blocked>
      %xa = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
      %yb = ttg.convert_layout %y : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
      %d = tt.dot %xa, %yb, %acc : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
      %na = tt.addptr %pa, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %nb = tt.addptr %pb, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %d, %na, %nb : tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return %r#0 : tensor<32x32xf32, #acc>
  }

  // A store in the body could alias what a moved load reads, so the loop
  // stays as it is.
  // CHECK-LABEL: @writes_memory
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  tt.func @writes_memory(%a: tensor<32x32x!tt.ptr<f32>, #blocked>, %b: tensor<32x32x!tt.ptr<f32>, #blocked>, %step: tensor<32x32xi32, #blocked>, %k: i32) -> tensor<32x32xf32, #acc> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #acc>
    %r:3 = scf.for %i = %c0 to %k step %c1 iter_args(%acc = %zero, %pa = %a, %pb = %b) -> (tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>)  : i32 {
      %x = tt.load %pa : tensor<32x32x!tt.ptr<f32>, #blocked>
      %y = tt.load %pb : tensor<32x32x!tt.ptr<f32>, #blocked>
      %xa = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opA>
      %yb = ttg.convert_layout %y : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #opB>
      %d = tt.dot %xa, %yb, %acc : tensor<32x32xf32, #opA> * tensor<32x32xf32, #opB> -> tensor<32x32xf32, #acc>
      tt.store %pb, %x : tensor<32x32x!tt.ptr<f32>, #blocked>
      %na = tt.addptr %pa, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %nb = tt.addptr %pb, %step : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %d, %na, %nb : tensor<32x32xf32, #acc>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    }
    tt.return %r#0 : tensor<32x32xf32, #acc>
  }
}
