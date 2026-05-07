_base_ = [
    "../_base_/datasets/json/video_ocv.py",  # dataset config
    "../_base_/models/e2espot.py",  # model config,
    "../_base_/schedules/e2e_100_map.py", #trainer config
]

work_dir = "outputs/custom_infer"

log_level = 'INFO'  # The level of logging

dali = False

dataset = dict(
    test = dict(
        path = "c:/apped/data/samples/test_5min.mp4",
        results = "results_spotting_test_ocv",
        dataloader = dict(
            num_workers = 0,
            batch_size = 4, # Smaller batch size just in case
        )
    )
)

visualizer = dict(
    threshold=0.2,
    annotation_range=5000,  # ms
    seconds_to_skip=30,
    scale=1.5,
)
