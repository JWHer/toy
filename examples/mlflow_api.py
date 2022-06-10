import mlflow
from mlflow.entities import ViewType

def create_experiment(exp_name="Test experiment"):
    """[summary]

    Args:
        exp_name ([str]): [experiment name]

    Returns:
        [mlflow.Experiment]: [experiment info]
    """    
    experiment_id = mlflow.create_experiment(exp_name)
    experiment = mlflow.get_experiment(experiment_id=experiment_id)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    return experiment


def get_experiment(exp_id):
    experiment = mlflow.get_experiment(exp_id)
    print("Name: {}".format(experiment.name))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    return experiment


def get_experiment_by_name(exp_name):
    experiment = mlflow.get_experiment_by_name(exp_name)
    print("Name: {}".format(experiment.name))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


def set_experiment(exp_name):
    # Set an experiment name, which must be unique and case sensitive.
    mlflow.set_experiment(exp_name)

    # Get Experiment Details
    # experiment = mlflow.get_experiment_by_name(exp_name)
    # print("Experiment_id: {}".format(experiment.experiment_id))
    # print("Artifact Location: {}".format(experiment.artifact_location))
    # print("Tags: {}".format(experiment.tags))
    # print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


def get_experiment_list():
    exp_list = mlflow.list_experiments()
    print(exp_list)


def delete_experiment(exp_id):
    """[summary]
        삭제지만, 완전한 삭제가 아닙니다. lifecycle_stage를 조회해 확인할 수 있습니다.
    Args:
        exp_id ([type]): [description]
    """    
    mlflow.delete_experiment(exp_id)


def get_run(run_id):
    #run_id = run.info.run_id
    run = mlflow.get_run(run_id)
    print("run_id: {}; lifecycle_stage: {}".format(run_id, run.info.lifecycle_stage))


def get_run_list(exp_id):
    # Create two runs
    with mlflow.start_run() as run1:
        mlflow.log_param("p", 0)

    with mlflow.start_run() as run2:
        mlflow.log_param("p", 1)

    # Delete the last run
    mlflow.delete_run(run2.info.run_id)

    def print_run_infos(run_infos):
        for r in run_infos:
            print("- run_id: {}, lifecycle_stage: {}".format(r.run_id, r.lifecycle_stage))

    print("Active runs:")
    print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.ACTIVE_ONLY))

    print("Deleted runs:")
    print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.DELETED_ONLY))

    print("All runs:")
    print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.ALL))


def delete_run(run_id):
    # run_id = run.info.run_id
    mlflow.delete_run(run_id)
    # print("run_id: {}; lifecycle_stage: {}".format(run_id,
    # mlflow.get_run(run_id).info.lifecycle_stage))


def start_run():
    # mlflow.start_run()
    # run = mlflow.active_run() # => 실행중인 run이 여러개면?
    # mlflow.log_param("somename", "value")
    # mlflow.log_metric(key=["keys"], value=["values"], step=0)
    # mlflow.end_run()

    with mlflow.start_run() as run:
        print("Active run_id: {}".format(run.info.run_id))
        mlflow.log_param("somename", "value2")
        mlflow.log_metric(key=["keys"], value=["values2"], step=1)


####################     look here     ####################

def execute_run(exp_id):
    mlflow.set_tracking_uri("http://mlflow:5000")
    # https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.run
    # project_uri = "file:///host/home/jwher/autocare_tx_system"
    # project_uri = "https://gitlab.com/snuailab/autocare_tx_system.git"
    project_uri = "https://github.com/mlflow/mlflow-example.git"
    # params = {
    #     "mode": "train",
    #     "exp_name": "phase0",
    #     "batch-size": "1",
    #     "weights": "/host/home/jwher/autocare_tx_system/model_farm/yolov4-p5.pt",
    #     "img-size": "64",
    #     "logdir": "runs_phase0",
    #     "data": "/host/home/jwher/autocare_tx_system/tx_model/ScaledYOLOv4/data/yolov4-p5.yaml",
    #     "hyp": "/host/home/jwher/autocare_tx_system/tx_model/ScaledYOLOv4/data/hyp.hpo0923.yaml",
    #     "train_id": "461a0240-7a25-4599-99e9-630c7b596399",
    #     "val_id": "8ea7c231-ba11-43c6-a96a-5cd4eb56d2f0",
    #     "classes": 0
    # }
    params = { "alpha":0.1, "l1_ratio":0.1}
    mlflow.run(project_uri,
        experiment_id=exp_id, 
        parameters=params, use_conda=False)


def set_tag(tags):
    """[summary]
    태그 설정(메타데이터 업데이트)에 사용

    Args:
        tags ([dict]): [description]
    """    
    tags = {"engineering": "ML Platform",
        "release.candidate": "RC1",
        "release.version": "2.2.0"}

    # Set a batch of tags
    with mlflow.start_run():
        mlflow.set_tag("release.version", "2.2.0")
        mlflow.set_tags(tags)


def miscellaneous():
    # use third-party logger
    mlflow.autolog(log_models=False, exclusive=True)
    import sklearn
    mlflow.sklearn.autolog(LogModel=True)

    # delete tag
    tags = {"engineering": "ML Platform",
        "engineering_remote": "ML Platform"}
    with mlflow.start_run() as run:
        mlflow.set_tags(tags)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.delete_tag("engineering_remote")

    # get artifact uri
    with mlflow.start_run():
        mlflow.log_artifact("features.txt", artifact_path="features")
    # Fetch the artifact uri root directory
    artifact_uri = mlflow.get_artifact_uri()
    print("Artifact uri: {}".format(artifact_uri))
    # Fetch a specific artifact uri
    artifact_uri = mlflow.get_artifact_uri(artifact_path="features/features.txt")
    print("Artifact uri: {}".format(artifact_uri))

    # get registry(model) uri
    mr_uri = mlflow.get_registry_uri()
    print("Current model registry uri: {}".format(mr_uri))
    # Get the current tracking uri
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    # They should be the same
    assert mr_uri == tracking_uri

    # get tracking uri
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    # set tracking uri
    mlflow.set_tracking_uri("file:///tmp/my_tracking")
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))


if __name__ == "__main__":
    # create_experiment()
    # get_run("2f2dff6a5dfc49068640c294c7dc4909")

    # execute_run(1)

    #get_experiment_list()

    # mlflow.set_experiment("Default")
    # mlflow.set_experiment("Hell")
    execute_run(1)

    pass