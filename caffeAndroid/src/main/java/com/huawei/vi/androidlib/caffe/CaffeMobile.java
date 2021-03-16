package com.huawei.vi.androidlib.caffe;

import org.opencv.core.Mat;

import android.util.Log;

/**
 * Created by Hongbing Li
 */
public class CaffeMobile {
	private static final String TAG = "CaffeMobile";
	
    private long mNativeObj = 0;
    
    static {
    	try {
    		System.loadLibrary("caffe_jni");
    		Log.w(TAG, "caffe_jni library loaded success");
    	} catch (UnsatisfiedLinkError e) {
    		Log.e(TAG, "caffe_jni library not found!");
    	}
    }

    public CaffeMobile(String modelPath, String weightsPath) {
        mNativeObj = nativeCreateObject(modelPath, weightsPath);
    }

    public void setMean(float[] meanValues) {
        setMeanWithMeanValues(mNativeObj, meanValues);
    }

    public void setMean(String meanFile) {
        setMeanWithMeanFile(mNativeObj, meanFile);
    }
    
    //0: resize, 1: center crop, 2: adaptive crop
    public void setCropType(int cropType) {
    	setCropType(mNativeObj, cropType);
    }
    
    public void setMapping(String labelFile, String indexMapFile) {
    	setMapping(mNativeObj, labelFile, indexMapFile);
    }

    public int[] predictImage(String imgPath, int k) {
        return predictImage(mNativeObj, imgPath, k);
    }

    public int[] predictImage(Mat rgbaMat, int k) {
        return predictImageByMat(mNativeObj, rgbaMat.getNativeObjAddr(), k);
    }
    
    public int[] getMappingScore(int k) {
    	return getMappingScore(mNativeObj, k);
    }
    
    public int[] predictImageWithMapping(String imgPath, int k) {
        return predictImageWithMapping(mNativeObj, imgPath, k);
    }

    public int[] predictImageWithMapping(Mat rgbaMat, int k) {
        return predictImageByMatWithMapping(mNativeObj, rgbaMat.getNativeObjAddr(), k);
    }

    public float[] getProbs(int k) {
        return getProbs(mNativeObj, k);
    }

    private native long nativeCreateObject(String modelPath, String weightsPath);

    public native void setNumThreads(int numThreads);

    public native void enableLog(boolean enabled);  // currently nonfunctional

    //public native int loadModel(String modelPath, String weightsPath);  // required

    private native void setMeanWithMeanFile(long thiz, String meanFile);

    private native void setMeanWithMeanValues(long thiz, float[] meanValues);

    private native void setScale(long thiz, float scale);
    
    private native void setCropType(long thiz, int cropType);
    
    private native void setMapping(long thiz, String labelFile, String indexMapFile);

    private native int[] predictImage(long thiz, String imgPath, int k);

    private native int[] predictImageByMat(long thiz, long imgRgba, int k);
    
    private native int[] getMappingScore(long thiz, int k);
    
    private native int[] predictImageWithMapping(long thiz, String imgPath, int k);

    private native int[] predictImageByMatWithMapping(long thiz, long imgRgba, int k);

    private native float[] getProbs(long thiz, int k);

    private native float[][] extractFeatures(long thiz, String imgPath, String blobNames);
}
