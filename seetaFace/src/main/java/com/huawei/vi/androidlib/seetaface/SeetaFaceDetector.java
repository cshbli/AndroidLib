package com.huawei.vi.androidlib.seetaface;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

public class SeetaFaceDetector {
	private long mNativeObj = 0;

    public SeetaFaceDetector(String modelPath) {
        mNativeObj = nativeCreateObject(modelPath);
    }
    
    public void release() {
    	nativeDestroyObject(mNativeObj);
    }
    
    public void detect(Mat image, MatOfRect faces) {
    	nativeDetect(mNativeObj, image.getNativeObjAddr(), faces.getNativeObjAddr());
    }
   
    private native long nativeCreateObject(String modelPath);
    private native void nativeDestroyObject(long thiz);
    private native void nativeDetect(long thiz, long inputImage, long faces);
}
