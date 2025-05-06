// <copyright file="LinearAlgebra.Matrix.fs" company="Math.NET">
// Math.NET Numerics, part of the Math.NET Project
// http://numerics.mathdotnet.com
// http://github.com/mathnet/mathnet-numerics
//
// Copyright (c) 2009-2014 Math.NET
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// </copyright>

namespace MathNet.Numerics.LinearAlgebra.GPU

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open Brahma.FSharp

/// A module which implements functional matrix operations.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Matrix =

    /// !!!
    let iter f (v: #Matrix<_>) = v

    /// !!!
    let map2 (f: Quotations.Expr<'a -> 'b -> 'c>) (v: #Matrix<'a>) (w: #Matrix<'b>) =
        let device = ClDevice.GetFirstAppropriateDevice() // ?
        let context = RuntimeContext(device).ClContext

        let n = v.RowCount
        let m = v.ColumnCount
        
        let m1_gpu = context.CreateClArray<_>(v.ToColumnMajorArray(), HostAccessMode.NotAccessible)
        let m2_gpu = context.CreateClArray<_>(w.ToColumnMajorArray(), HostAccessMode.NotAccessible)
        let m3_gpu =
            context.CreateClArray(
                n * m,
                HostAccessMode.NotAccessible,
                deviceAccessMode=DeviceAccessMode.WriteOnly,
                allocationMode = AllocationMode.Default
            )

        let localWorkSize = 4

        let kernel =
            <@
                fun (r: Range2D) (m1: ClArray<_>) (m2: ClArray<_>) (m3: ClArray<_>)  ->
                    let i = r.GlobalID0
                    let j = r.GlobalID1

                    m3.[i * m + j] <- (%f) m1[i * m + j] m2[i * m + j]
            @>

        let kernel = context.Compile kernel

        let ndRange =
            Range2D(
                n,
                m,
                localWorkSize,
                localWorkSize
            )

        let commandQueue = context.QueueProvider.CreateQueue()
        let kernel = kernel.GetKernel()

        commandQueue.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange m1_gpu m2_gpu m3_gpu))
        commandQueue.Post(Msg.CreateRunMsg<_, _> kernel)
        let result : 'c[] = Array.zeroCreate(n * m)
        let result = commandQueue.PostAndReply(fun ch -> Msg.CreateToHostMsg(m3_gpu, result, ch))
        commandQueue.Post(Msg.CreateFreeMsg m1_gpu)
            
        commandQueue.Post(Msg.CreateFreeMsg m2_gpu)
            
        commandQueue.Post(Msg.CreateFreeMsg m3_gpu)

        result |> DenseMatrix.raw n m

    /// !!!
    let mXm (v: #Matrix<_>) (w: #Matrix<_>) = v

