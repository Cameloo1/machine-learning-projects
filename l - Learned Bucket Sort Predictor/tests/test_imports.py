def test_project_modules_import():
    import learned_bucket_sort
    import learned_bucket_sort.amortized_benchmark
    import learned_bucket_sort.baseline
    import learned_bucket_sort.benchmark
    import learned_bucket_sort.cdf_model
    import learned_bucket_sort.data
    import learned_bucket_sort.learned_sort
    import learned_bucket_sort.metrics
    import learned_bucket_sort.part5_evidence
    import learned_bucket_sort.scale_closure
    import learned_bucket_sort.scenarios
    import learned_bucket_sort.torch_mlp_cdf

    assert learned_bucket_sort.__all__ == [
        "amortized_benchmark",
        "baseline",
        "benchmark",
        "cdf_model",
        "data",
        "learned_sort",
        "metrics",
        "part5_evidence",
        "scale_closure",
        "scenarios",
        "torch_mlp_cdf",
    ]
