open System
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Reports
open Perfolizer.Horology

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics

type Benchmark() =

    static member Sizes = [|2|]


    [<ParamsSource("Sizes")>]
    member val MatrixSize = 0 with get, set
    member val MatrixToSort = Matrix<double>.Build.Random(1, 1) with get, set

    [<IterationSetup>]
    member this.GetMatrixToSort () = 
        this.MatrixToSort <- Matrix<double>.Build.Random(this.MatrixSize, this.MatrixSize)

    [<Benchmark>]
    member this.iter () = Matrix.iter (fun x -> x |> ignore) this.MatrixToSort

    [<Benchmark>]
    member this.iterGPU () = GPU.Matrix.iter <@ fun x -> x |> ignore @> this.MatrixToSort


module Main =
    [<EntryPoint>]
    let main argv =

        let config = ManualConfig.Create(DefaultConfig.Instance)
                        .WithSummaryStyle(SummaryStyle.Default.WithTimeUnit(TimeUnit.Millisecond)) 
                        .WithOptions(ConfigOptions.DisableOptimizationsValidator) 

        let benchmarks =
            BenchmarkSwitcher [| typeof<Benchmark> |]

        benchmarks.RunAll(config, argv) |> ignore
        0