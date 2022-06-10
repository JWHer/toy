def arg_to_json():

    args = ["--mode=train",
            "--exp_name=phase2",
            "--batch-size=1",
            "--weights=/host/home/jwher/autocare_tx_system/model_farm/yolov4-p5.pt",
            "--img-size=64",
            "--logdir=runs_demo",
            "--data=/host/home/jwher/autocare_tx_system/tx_model/ScaledYOLOv4/data/yolov4-p5.yaml",
            "--hyp=/host/home/jwher/autocare_tx_system/tx_model/ScaledYOLOv4/data/hyp.hpo0923.yaml",
            "--train_id=461a0240-7a25-4599-99e9-630c7b596399",
            "--val_id=8ea7c231-ba11-43c6-a96a-5cd4eb56d2f0",
            "--classes=wheelchair blind stroller person",

            "--resume=False",
            "--u_mixup=0.5",
            "--teacher_ckpt=/host/home/jwher/autocare_tx_system/mlruns/0/76f74b6fdd8a4879b1fc6074ff766d8f/artifacts/weights/best.pt",
            "--student_model=p5",
            "--u_train_id=2cc500f0-5906-41fe-af63-c47368e1f296 3d160a84-5792-4d05-a365-30e2144d8151",
            "--SSL",
            "--u_mosaic",
            "--augment"]