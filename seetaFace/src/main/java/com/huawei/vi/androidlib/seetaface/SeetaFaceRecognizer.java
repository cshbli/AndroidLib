package com.huawei.vi.androidlib.seetaface;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

public class SeetaFaceRecognizer {
	private static final String TAG = "SeetaFaceRecognizer";
	
	// Scale image to small size for fast face detection
	private static final int IMAGE_WIDTH = 640;
	private static final int IMAGE_HEIGHT = 480;
	
	private long mDetectionObj = 0;
	private long mAlignmentObj = 0;
	private long mRecognitionObj = 0;
	
	private float[] features;
	private Rect[] faces;
	
	static {
        System.loadLibrary("viplnet_jni");
    }

    public SeetaFaceRecognizer(String detectionModel, String alignmentModel, String recognitionModel) {
        mDetectionObj = nativeCreateDetectionObject(detectionModel);
        mAlignmentObj = nativeCreateAlignmentObject(alignmentModel);
        mRecognitionObj = nativeCreateRecognitionObject(recognitionModel);
    }
    
    public void release() {
    	nativeDestroyDetectionObject(mDetectionObj);
    	nativeDestroyAlignmentObject(mAlignmentObj);
    	nativeDestroyRecognitionObject(mRecognitionObj);
    }
    
    public float[] extractFeatures(Mat image) {
    	return nativeExtractFeatures(mDetectionObj, mAlignmentObj, mRecognitionObj, image.getNativeObjAddr());
    }
    
    public float calcSimilarity(float[] features1, float[] features2) {
    	return nativeCalcSimilarity(mRecognitionObj, features1, features2);
    }
    
    public synchronized int detectFacesAndExtractFeatures(Bitmap bmp) {
    	Bitmap resizedBmp;
    	int scale = 1;
    	scale = Math.min(bmp.getWidth() / IMAGE_WIDTH, bmp.getHeight() / IMAGE_HEIGHT);
    	scale = Math.max(scale, 1);
    	Log.i(TAG, "Image scale: " + scale);
    	resizedBmp = Bitmap.createScaledBitmap(bmp, bmp.getWidth() / scale, bmp.getHeight() / scale, true);
        
        // Convert the bitmap to opencv Mat structure
        Bitmap bmp32 = resizedBmp.copy(Bitmap.Config.ARGB_8888, true);
        Mat matRGBA = new Mat(bmp32.getHeight(), bmp32.getWidth(), CvType.CV_8UC4);
        Utils.bitmapToMat(bmp32, matRGBA);
        
        long startTime = SystemClock.uptimeMillis();
        MatOfRect faceMat = new MatOfRect();        
        features = nativeDetectFacesAndExtractFeatures(mDetectionObj, mAlignmentObj, mRecognitionObj, 
        		matRGBA.getNativeObjAddr(), faceMat.getNativeObjAddr());          
        Log.w(TAG, String.format("Face detection and feature extraction elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));
        
        if (features == null || faceMat.empty()) {
        	Log.w(TAG, "No faces were detected.");
        	return 0;
        }
        
        faces = faceMat.toArray();
        for (int i = 0; i < faces.length; i++) {
        	faces[i].x = faces[i].x * scale;
        	faces[i].y = faces[i].y * scale;
        	faces[i].width = faces[i].width * scale;
        	faces[i].height = faces[i].height * scale;
        }

        return faces.length;
    }
    
    public float[] getFaceFeatures() {
    	return features;
    }
    
    public Rect[] getDetectedFaces() {
    	return faces;
    }
   
    private native long nativeCreateDetectionObject(String detectionModel);
    private native long nativeCreateAlignmentObject(String alignmentModel);
    private native long nativeCreateRecognitionObject(String recognitionModel);
    private native void nativeDestroyDetectionObject(long detectionObj);
    private native void nativeDestroyAlignmentObject(long alignmentObj);
    private native void nativeDestroyRecognitionObject(long recognitionObj);
    private native float[] nativeExtractFeatures(long detectionObj, long alignmentObj, long recognitionObj, long jImage);
    private native float nativeCalcSimilarity(long recognitionObj, float[] features1, float[] features2);
    private native float[] nativeDetectFacesAndExtractFeatures(long detectionObj, long alignmentObj, long recognitionObj, long jImage, long faces);
}
