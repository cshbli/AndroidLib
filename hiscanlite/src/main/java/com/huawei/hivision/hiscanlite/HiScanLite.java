package com.huawei.hivision.hiscanlite;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by hongbing on 1/24/18.
 */

public class HiScanLite {
    private static final String TAG = "HiScanLite";

    // Adding one category of "person" for people without face showing up
    private String[] labels_en = {"people", "commodity", "flower", "pet", "logo", "automobile", "building", "store sign", "document", "business card", "code", "art", "green plant", "background", "person"};
    private String[] labels_cn = {"人脸", "商品", "花卉", "宠物", "Logo", "汽车", "建筑景点", "店招", "文本", "名片", "码", "艺术品", "绿植", "背景", "人物"};
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

    private ImageClassifier classifier = null;

    /**
     * Initializes a hiscan session for classifying images.
     *
     * @param myContext The application context to be used to load assets.
     * @throws IOException
     */
    public HiScanLite(Context myContext) {
        labelMap = new HashMap<String, String>();
        for (int i = 0; i < labels_en.length; i++) {
            labelMap.put(labels_en[i], labels_cn[i]);
        }

        try {
            // create either a new ImageClassifierQuantizedMobileNet or an ImageClassifierFloatInception
            //classifier = new ImageClassifierQuantizedMobileNet(getActivity());
            classifier = new ImageClassifierFloatInception((Activity)myContext);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.", e);
        }
    }

    private int getIndexOfArrayList(List<Recognition> inputs, String label) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i).getTitle().startsWith(label))
                return i;
        }
        return -1;
    }

    private void adjustScoringResults(float[] scores) {
        float[] final_scores = new float[labels_en.length];

        System.arraycopy(scores, 0, final_scores, 0, labels_en.length);

        // For people, raise probability of automobile, building, flower, green plant, commodity, pet,
        // as they are together a lot
        /* This has been handled by multi-label categories inside the model
        if (scores[idx_people] >= RECOGNITION_THRESHOLD || scores[idx_person] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_automobile] += 0.25;
            final_scores[idx_building] += 0.25;
            final_scores[idx_commodity] += 0.2;
            final_scores[idx_flower] += 0.25;
            final_scores[idx_green_plant] += 0.2;
            final_scores[idx_pet] += 0.25;
        }
        */

        // For flower, raise "green plant"
        if (scores[idx_flower] >= RECOGNITION_THRESHOLD) {
            final_scores[idx_green_plant] += 0.2;
        }

        // raise people by 0.25
        /* This has been handled by multi-label categories inside the model
        if (scores[idx_person] < 0.3 && (scores[idx_automobile] >= RECOGNITION_THRESHOLD ||
            scores[idx_building] >= RECOGNITION_THRESHOLD ||
            scores[idx_commodity] >= RECOGNITION_THRESHOLD ||
            scores[idx_flower] >= RECOGNITION_THRESHOLD ||
            scores[idx_green_plant] >= RECOGNITION_THRESHOLD ||
            scores[idx_pet] >= RECOGNITION_THRESHOLD)) {
            final_scores[idx_people] += 0.25;
        }
        */

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


    /**
     * Classify an input image
     *
     * @param bmp the input image bitmap
     * @return A list of label and confidence pair, can be null if the image not belongs to one the 13 top classes
     */
    public void recognizeImage(final Bitmap bmp) {
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

        String textToShow = classifier.classifyFrame(resizedBmp);

        Log.e(TAG, textToShow);


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
    public void getTfResults() {}

    /**
     * Closes the hiscan session
     */
    public void close() {

    }
}
