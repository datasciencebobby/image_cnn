from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"..\runs\detect\train-5\weights\best.pt")

    results = model.train(data=r".\data\data.yaml", epochs=20)

    results = model.val()