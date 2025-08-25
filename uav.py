import os
import shutil
import time
from pathlib import Path
import gc
import cv2
import yaml
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torch2trt import torch2trt

def prepareDataset():
    print("Choose dataset folder:")
    print("1. DSC_6301")
    print("2. DSC_6302")
    print("3. DSC_6303")

    choice = input("Enter choice (1-3): ").strip()
    basePath = r"./AUTH_UAV_Small"

    if choice == "1":
        sourceFolders = [os.path.join(basePath, "DSC_6301")]
        datasetName = "UAV_Dataset_6301"
    elif choice == "2":
        sourceFolders = [os.path.join(basePath, "DSC_6302")]
        datasetName = "UAV_Dataset_6302"
    elif choice == "3":
        sourceFolders = [os.path.join(basePath, "DSC_6303")]
        datasetName = "UAV_Dataset_6303"
    else:
        print("Invalid choice, using DSC_6301")
        sourceFolders = [os.path.join(basePath, "DSC_6301")]
        datasetName = "UAV_Dataset_6301"

    print(f"Selected: {datasetName}")
    outputDataset = datasetName
    datasetPath = Path(outputDataset)

    for split in ['train', 'val', 'test']:
        (datasetPath / 'images' / split).mkdir(parents=True, exist_ok=True)
        (datasetPath / 'labels' / split).mkdir(parents=True, exist_ok=True)

    allFiles = []
    for folder in sourceFolders:
        folderPath = Path(folder)
        if folderPath.exists():
            imageFiles = list(folderPath.glob('*.png'))
            allFiles.extend(imageFiles)
            print(f"Found {len(imageFiles)} images in {folder}")

    totalFiles = len(allFiles)
    trainCount = int(totalFiles * 0.5)
    valCount = int(totalFiles * 0.2)

    trainFiles = allFiles[:trainCount]
    valFiles = allFiles[trainCount:trainCount + valCount]
    testFiles = allFiles[trainCount + valCount:]

    print(f"Split sizes -> Train: {len(trainFiles)}, Val: {len(valFiles)}, Test: {len(testFiles)}")
    splits = {'train': trainFiles, 'val': valFiles, 'test': testFiles}

    for splitName, fileList in splits.items():
        for i, imgFile in enumerate(fileList):
            newImgName = f"{splitName}_{i:06d}.png"
            shutil.copy2(imgFile, datasetPath / 'images' / splitName / newImgName)
            originalLabelFile = imgFile.with_suffix('.txt')
            newLabelFile = datasetPath / 'labels' / splitName / f"{splitName}_{i:06d}.txt"
            if originalLabelFile.exists():
                shutil.copy2(originalLabelFile, newLabelFile)
            else:
                print(f"Warning: Label file missing for image {imgFile.name} - skipping label")
                newLabelFile.touch()

    config = {
        'path': str(datasetPath.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {0: 'UAV'}
    }

    yamlPath = datasetPath / 'data.yaml'
    with open(yamlPath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(yamlPath), datasetName

def main():
    yamlPath, datasetName = prepareDataset()

    print("\nChoose TensorRT precision:")
    print("1. FP32")
    print("2. FP16")
    print("3. INT8")
    precisionChoice = input("Enter choice (1-3): ").strip()

    precisionMap = {
        '1': 'fp32',
        '2': 'fp16',
        '3': 'int8'
    }

    precision = precisionMap.get(precisionChoice)
    if not precision:
        print("Invalid choice. Defaulting to FP32.")
        precision = 'fp32'

    print(f"\nSelected Precision: {precision.upper()}")

    weightPath = trainModel(yamlPath)
    trtModel = optimizeModel(weightPath, precision) 
    testModelPerformance(yamlPath, weightPath, trtModel, precision)

def trainModel(yamlPath):
    model = YOLO('yolo11m.pt')
    results = model.train(
        data=yamlPath,
        epochs=50,
        patience=30,
        imgsz=640,
        batch=8,
        device="0" if torch.cuda.is_available() else "cpu",
        save=True,
        plots=True,
        amp=True
    )
    
    defaultPath = "runs/detect/train/weights/best.pt"
    if os.path.exists(defaultPath):
        return defaultPath
    else:
        runsDir = Path("runs/detect")
        if runsDir.exists():
            trainDirs = [d for d in runsDir.iterdir() if d.is_dir() and d.name.startswith("train")]
            if trainDirs:
                latestDir = max(trainDirs, key=lambda x: x.stat().st_mtime)
                bestPath = latestDir / 'weights' / 'best.pt'
                if bestPath.exists():
                    print(f"Found weights at: {bestPath}")
                    return str(bestPath)
        
        print("Warning: best.pt not found, using last.pt or model weights")
        lastPath = 'runs/detect/train/weights/last.pt'
        if os.path.exists(lastPath):
            return lastPath
        else:
            fallbackPath = 'current_model.pt'
            model.save(fallbackPath)
            return fallbackPath

def optimizeModel(modelPath, precision='fp32'):
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found: {modelPath}")
    
    print(f"Loading model from: {modelPath}")
    
    yoloModel = YOLO(modelPath)
    model = yoloModel.model.eval().cuda()
    
    dummyInput = torch.randn(1, 3, 640, 640).cuda()

    print(f"Converting to TensorRT with {precision} precision...")
    
    try:
        if precision == 'fp16':
            trtModel = torch2trt(model, [dummyInput], fp16_mode=True)
        elif precision == 'int8':
            trtModel = torch2trt(model, [dummyInput], int8_mode=True)
        else:
            trtModel = torch2trt(model, [dummyInput])

        outputPath = f"optimized_{precision}.pth"
        torch.save(trtModel.state_dict(), outputPath)
        print(f"Optimized model saved to: {outputPath}")
        return trtModel
        
    except Exception as e:
        print(f"Error during TensorRT conversion: {e}")
        print("Returning original model...")
        return model

def testModelPerformance(yamlPath, originalModelPath, trtModel, precision):    
    with open(yamlPath, 'r') as f:
        dataConfig = yaml.safe_load(f)
    
    testImagesPath = Path(dataConfig['path']) / 'images' / 'test'
    testImages = list(testImagesPath.glob('*.png'))
    
    if not testImages:
        return
    
    print(f"\nTesting on {len(testImages)} test images")
    
    print("\nTesting Original Model")
    originalModel = YOLO(originalModelPath)
    originalMetrics = originalModel.val(data=yamlPath)
    originalFPS = measureFPSOnImages(originalModel, testImages, "Original")
    
    print(f"\nTesting TensorRT {precision.upper()} Model")
    trtFPS = measureFPSOnTensorRT(trtModel, testImages, f"TensorRT {precision.upper()}")
    
    print(f"\nPerformance Comparison")
    print(f"Original Model:")
    print(f"  - mAP@50: {originalMetrics.box.map50:.3f}")
    print(f"  - mAP@50-95: {originalMetrics.box.map:.3f}")
    print(f"  - Precision: {originalMetrics.box.mp:.3f}")
    print(f"  - Recall: {originalMetrics.box.mr:.3f}")
    print(f"  - Average FPS: {originalFPS:.2f}")
    
    print(f"\nTensorRT {precision.upper()} Model:")
    print(f"  - Average FPS: {trtFPS:.2f}")
    print(f"  - Speed improvement: {trtFPS/originalFPS:.2f}x faster")
    
    createComparisonPlot(originalMetrics, originalFPS, trtFPS, precision)

def qualitativeDetections(model, imagePaths, max_images=5):
    print(f"Saving qualitative detection samples ({max_images} images)")
    for i, imgPath in enumerate(imagePaths[:max_images]):
        img = cv2.imread(str(imgPath))
        if img is None:
            continue
        results = model(img)
        ImgWPlot = results[0].plot()
        savePath = f"image_{i}.png"
        cv2.imwrite(savePath, ImgWPlot)
        print(f"Saved {savePath}")

def measureFPSOnImages(model, imageList, modelName):
    frameCount = 0
    totalTime = 0
    
    print(f"Measuring FPS for {modelName}...")
    
    for i, imgPath in enumerate(imageList):
        img = cv2.imread(str(imgPath))
        if img is None:
            continue
            
        start = time.time()
        _ = model(img, verbose=False)
        elapsed = time.time() - start
        totalTime += elapsed
        frameCount += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(imageList)} images...")
    
    if totalTime > 0 and frameCount > 0:
        avgFps = frameCount / totalTime
        print(f"{modelName} - Processed {frameCount} images in {totalTime:.4f} seconds")
        print(f"{modelName} - Average FPS: {avgFps:.2f}")
        return avgFps
    else:
        print(f"Error measuring FPS for {modelName}")
        return 0.0
    

def measureFPSOnTensorRT(trtModel, imageList, modelName):
    frameCount = 0
    totalTime = 0
    
    print(f"Measuring FPS for {modelName}...")
    
    for i, imgPath in enumerate(imageList):
        img = cv2.imread(str(imgPath))
        if img is None:
            continue
            
        img = cv2.resize(img, (640, 640))
        imgTensor = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.0
        imgTensor = imgTensor.unsqueeze(0)
        
        start = time.time()
        with torch.no_grad():
            _ = trtModel(imgTensor)
        elapsed = time.time() - start
        totalTime += elapsed
        frameCount += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(imageList)} images...")
    
    if totalTime > 0 and frameCount > 0:
        avgFps = frameCount / totalTime
        print(f"{modelName}: Processed {frameCount} images in {totalTime:.4f} seconds")
        print(f"{modelName}: Average FPS: {avgFps:.2f}")
        return avgFps
    else:
        print(f"Error measuring FPS for {modelName}")
        return 0.0

def createComparisonPlot(originalMetrics, originalFPS, trtFPS, precision):
        
    metricsData = {
        'mAP@50': originalMetrics.box.map50,
        'mAP@50-95': originalMetrics.box.map,
        'Precision': originalMetrics.box.mp,
        'Recall': originalMetrics.box.mr
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(metricsData.keys(), metricsData.values(), 
            color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    ax1.set_ylim(0, 1)
    ax1.set_title("Model Evaluation Metrics (Original)")
    ax1.set_ylabel("Score")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, (key, value) in enumerate(metricsData.items()):
        ax1.text(i, value + 0.02, f"{value:.3f}", ha='center', fontsize=10)
    
    models = ['Original', f'TensorRT {precision.upper()}']
    fpsValues = [originalFPS, trtFPS]
    colors = ['lightcoral', 'lightblue']
    
    bars = ax2.bar(models, fpsValues, color=colors)
    ax2.set_title("FPS Comparison")
    ax2.set_ylabel("FPS")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, fps in zip(bars, fpsValues):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{fps:.2f}', ha='center', va='bottom', fontsize=12)
    
    speedup = trtFPS / originalFPS if originalFPS > 0 else 0
    ax2.text(0.5, max(fpsValues) * 0.8, f'{speedup:.2f}x faster',
             ha='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
 
    
    plt.tight_layout()
    plt.savefig(f"tensorrt_comparison_{precision}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved as: tensorrt_comparison_{precision}.png")

def clean_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

if __name__ == '__main__':
    main()