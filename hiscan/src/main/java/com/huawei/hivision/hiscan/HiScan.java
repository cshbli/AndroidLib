package com.huawei.hivision.hiscan;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.util.Log;

import org.tensorflow.demo.Classifier;
import org.tensorflow.demo.TensorFlowImageClassifier;
import org.tensorflow.demo.env.ImageUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by hongbing on 1/24/18.
 */

public class HiScan {
    private static final String TAG = "HiScan";

    // Adding one category of "person" for people without face showing up
    private String[] labels_en = {"people", "commodity", "flower", "pet", "logo", "automobile", "building", "store sign", "document", "business card", "code", "art", "green plant", "background", "person", "food"};
    private String[] labels_cn = {"人脸", "商品", "花卉", "宠物", "Logo", "汽车", "建筑景点", "店招", "文本", "名片", "码", "艺术品", "绿植", "背景", "人物", "食物"};
    private static final String PREFIX_MULTI_LABEL = "multilabel";
    private int idx_people = 0;
    private int idx_commodity = 1;
    private int idx_flower = 2;
    private int idx_pet = 3;
    private int idx_logo = 4;
    private int idx_automobile = 5;
    private int idx_building = 6;
    private int idx_store_sign = 7;
    private int idx_document = 8;
    private int idx_business_card = 9;
    private int idx_code = 10;
    private int idx_art = 11;
    private int idx_green_plant = 12;
    private int idx_background = 13;
    private int idx_person = 14;
    private int idx_food = 15;

    Map<String, String> labelMap;

    private static final float RECOGNITION_THRESHOLD = 0.4f;

    /**
     * An immutable result returned by a hiscan describing what was recognized.
     */
    public class Recognition {
        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;


        public Recognition(final String title, final Float confidence) {
            this.title = title;
            this.confidence = confidence;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }
    }

    private int inception_input_size = 224;
    private Classifier tfClassifier = null;
    private List<Classifier.Recognition> tfResults;

    /**
     * Initializes a hiscan session for classifying images.
     *
     * @param myContext The application context to be used to load assets.
     * @throws IOException
     */
    public HiScan(Context myContext) {
        tfClassifier =
                TensorFlowImageClassifier.create(
                        myContext.getAssets(),
                        "file:///android_asset/scanner_mobilenet_0.50_224_optimized.pb", //MODEL_FILE,
                        "file:///android_asset/scanner_mobilenet_0.50_224_labels.txt", //LABEL_FILE,
                        inception_input_size, //INCEPTION_INPUT_SIZE,
                        128, //IMAGE_MEAN
                        128.0f, //IMAGE_STD,
                        "input", //INPUT_NAME,
                        "final_result"); //OUTPUT_NAME);

        labelMap = new HashMap<String, String>();
        for (int i = 0; i < labels_en.length; i++) {
            labelMap.put(labels_en[i], labels_cn[i]);
        }
    }

    private int getIndexOfArrayList(List<Recognition> inputs, String label) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i).getTitle().startsWith(label))
                return i;
        }
        return -1;
    }

    private void updateArrayList(List<Recognition> results, String labelPrefix, Classifier.Recognition recog) {
        int index = getIndexOfArrayList(results, labelPrefix);
        if (index < 0) {
            // this is the first appearance
            results.add(new Recognition(labelPrefix, recog.getConfidence()));
        }
        else {
            // update the existing item confidence
            Float prevConfidence = results.get(index).getConfidence();
            results.set(index, new Recognition(labelPrefix, prevConfidence + recog.getConfidence()));
        }
    }

    private void adjustScoringResults(float[] scores) {
        float[] final_scores = new float[labels_en.length];

        System.arraycopy(scores, 0, final_scores, 0, labels_en.length);

        // Adding flower and green plant together
        if (scores[idx_flower] >= scores[idx_green_plant]) {
            final_scores[idx_flower] += scores[idx_green_plant];
        } else {
            final_scores[idx_green_plant] += scores[idx_flower];
        }

        // For people, raise probability of automobile, building, flower, green plant, commodity, pet,
        // as they are together a lot
        if (scores[idx_people] >= RECOGNITION_THRESHOLD || scores[idx_person] >= RECOGNITION_THRESHOLD) {
            // Commodity and automobile have been handled by multi-label categories inside the model
            //final_scores[idx_automobile] += 0.25;
            //final_scores[idx_commodity] += 0.2;
            // 04/27: model removed handling multi-label categories about building, flower, green_plant and pet
            final_scores[idx_building] += 0.25;
            final_scores[idx_flower] += 0.25;
            final_scores[idx_green_plant] += 0.2;
            final_scores[idx_pet] += 0.25;
        }

        // For flower, raise "green plant"
        /*
        if (scores[idx_flower] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_green_plant] += 0.2;
        }
        */

        // raise people by 0.2
        if ((scores[idx_automobile] >= RECOGNITION_THRESHOLD ||
            scores[idx_building] >= RECOGNITION_THRESHOLD ||
            scores[idx_commodity] >= RECOGNITION_THRESHOLD ||
            scores[idx_flower] >= RECOGNITION_THRESHOLD ||
            scores[idx_green_plant] >= RECOGNITION_THRESHOLD ||
            scores[idx_pet] >= RECOGNITION_THRESHOLD) &&
            (scores[idx_person] < scores[idx_people] &&
             scores[idx_people] < RECOGNITION_THRESHOLD)) {
            final_scores[idx_people] += 0.2;
        }

        // For store sign, raise "building" and "logo"
        if (scores[idx_store_sign] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_building] += 0.2;
            final_scores[idx_logo] += 0.2;
        }

        // For building, raise "store sign" a little bit
        if (scores[idx_building] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_store_sign] += 0.1;
        }

        // For commodity, raise "Logo" a little bit
        if (scores[idx_commodity] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_logo] += 0.2;
        }

        // For Logo, raise "commodity" a little bit
        if (scores[idx_logo] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_commodity] += 0.2;
        }

        // For business card, lower "document" a little bit
        if (scores[idx_business_card] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_document] -= 0.2;
        }

        // if background >= 0.3, reduce the probability of all others except people
        if (scores[idx_background] >= 0.3) {
            for (int idx = 0; idx < labels_en.length; idx++) {
                if (idx != idx_background && idx != idx_people && idx != idx_person) {
                    final_scores[idx] -= 0.2;
                }
            }
        }

        System.arraycopy(final_scores, 0, scores, 0, labels_en.length);
    }

    private List<Recognition> labelMapping(List<Classifier.Recognition> inputs) {
        if (inputs != null) {
            float[] scores = new float[labels_en.length];

            for (final Classifier.Recognition recog : inputs) {
                for (int i = 0; i < labels_en.length; i++) {
                    if (recog.getTitle().startsWith(labels_en[i])) {
                        scores[i] += recog.getConfidence();
                        break;
                    }
                    // For multi-labels
                    else if (recog.getTitle().startsWith(PREFIX_MULTI_LABEL) && recog.getTitle().contains(labels_en[i])) {
                        scores[i] += recog.getConfidence();
                    }
                }
            }

            // Adjust the scores
            adjustScoringResults(scores);

            List<Recognition> finalResults = new ArrayList<Recognition>();

            for (int idx = 0; idx < labels_en.length; idx++) {
                if (scores[idx] >= RECOGNITION_THRESHOLD && idx != idx_background && idx != idx_person) {
                    finalResults.add(new Recognition(labels_en[idx], scores[idx]));
                }
            }

            if (finalResults.size() > 0) {
                /*
                for (int i = 0; i < finalResults.size(); i++) {
                    Log.e(TAG, finalResults.get(i).getTitle() + ": " + finalResults.get(i).getConfidence());
                }
                */
                return finalResults;
            }
            else {
                return null;
            }
        }

        return null;
    }

    /**
     * Classify an input image
     *
     * @param bmp the input image bitmap
     * @return A list of label and confidence pair, can be null if the image not belongs to one the 13 top classes
     */
    public List<HiScan.Recognition> recognizeImage(final Bitmap bmp) {
        // This getTransformationMatrix with "mainAspectRatio = True" caused some problems as training model always scale the
        // original image to the model input size without maintain aspect ratio
        /*
        Bitmap croppedBitmap = Bitmap.createBitmap(inception_input_size, inception_input_size, Bitmap.Config.ARGB_8888);
        Matrix cropTransform =
                ImageUtils.getTransformationMatrix(
                        bmp.getWidth(), bmp.getHeight(),
                        inception_input_size, inception_input_size,
                        0, true);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(bmp, cropTransform, null);

        tfResults = tfClassifier.recognizeImage(croppedBitmap);
        */


        // We will use simple Bitmap.createScaledBitmap without cropping and maintaining the original aspect ratio
        Bitmap resizedBmp = Bitmap.createScaledBitmap(bmp, inception_input_size, inception_input_size, true);

        tfResults = tfClassifier.recognizeImage(resizedBmp);

        return labelMapping(tfResults);
    }

    /**
     *
     * @param label English label
     * @return Chinese label
     */
    public String getChineseLabel(String label) {
        return labelMap.get(label);
    }

    /**
     * For debugging purpose only: Get internal raw recognition result after calling recognizeImage
     *
     * @return A list of label and confidence pair
     */
    public List<Classifier.Recognition> getTfResults() {return tfResults;}

    /**
     * Closes the hiscan session
     */
    public void close() {
        if (tfClassifier != null) {
            tfClassifier.close();
        }
    }
}
