package com.example.applayout.FederatedLearning;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.applayout.MainActivity;
import com.example.applayout.R;
import com.example.applayout.Report.ReportActivity;
import com.example.applayout.Report.ReportActivity2;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FederatedLearningActivity extends AppCompatActivity{
    Context context = this;
    ProgressBar progressBar;
    TextView status, text;
    Button btConfirm;

    private class TrainModelTask extends AsyncTask<Void,Integer,Void>{
        protected void onPreExecute(){
            super.onPreExecute();
            progressBar = findViewById(R.id.progressBar);
            progressBar.setVisibility(View.VISIBLE);
        }

        protected Void doInBackground(Void... voids){
            try (Interpreter anotherInterpreter = new Interpreter(loadModelFile(context.getAssets(),"model.tflite"))) {
                int NUM_EPOCHS = 100;
                int BATCH_SIZE = 100;
                int IMG_HEIGHT = 28;
                int IMG_WIDTH = 28;
                int NUM_TRAININGS = 60000;
                int NUM_BATCHES = NUM_TRAININGS / BATCH_SIZE;

                //List<FloatBuffer> trainImageBatches = new ArrayList<>(NUM_BATCHES);
                //List<FloatBuffer> trainLabelBatches = new ArrayList<>(NUM_BATCHES);
                List<FloatBuffer> trainImageBatches = new ArrayList<>(10);
                List<FloatBuffer> trainLabelBatches = new ArrayList<>(10);

                for (int i = 0; i < 10; ++i) {
                    String imagePath = "test_images/image" + i + ".png";
                    String labelPath = "labels/label" + i + ".txt";
                    FloatBuffer trainImages = readImageAsFloatBuffer(context, imagePath, IMG_WIDTH, IMG_HEIGHT);
                    FloatBuffer trainLabels = readLabelAsFloatBuffer(context, labelPath, 10); // Assuming 10 classes

                    if (trainImages != null && trainLabels != null) {
                        trainImageBatches.add(trainImages);
                        trainLabelBatches.add(trainLabels);
                    } else {
                        Log.e("FederatedLearning", "Failed to read image or label for batch " + i);
                    }
                }

                // Prepare training batches.
                /*for (int i = 0; i < NUM_BATCHES; ++i) {
                    ByteBuffer trainImageBuffer = ByteBuffer.allocateDirect(4 * IMG_HEIGHT * IMG_WIDTH).order(ByteOrder.nativeOrder());
                    FloatBuffer trainImages = trainImageBuffer.asFloatBuffer();

                    ByteBuffer trainLabelsBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                    FloatBuffer trainLabels = trainLabelsBuffer.asFloatBuffer();

                    // Fill the data values...
                    trainImageBatches.add((FloatBuffer) trainImages.rewind());
                    trainLabelBatches.add((FloatBuffer) trainLabels.rewind());
                }*/

                // Run training for a few steps.
                float[] losses = new float[NUM_EPOCHS];
                for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
                    for (int batchIdx = 0; batchIdx < 10; ++batchIdx) {
                        Map<String, Object> inputs = new HashMap<>();
                        inputs.put("x", trainImageBatches.get(batchIdx));
                        inputs.put("y", trainLabelBatches.get(batchIdx));

                        Map<String, Object> outputs = new HashMap<>();
                        FloatBuffer lossBuffer = FloatBuffer.allocate(1);
                        outputs.put("loss", lossBuffer);

                        anotherInterpreter.runSignature(inputs, outputs, "train");
                        final int progressPercentage = (epoch * 100) / NUM_EPOCHS;
                        progressBar.setProgress(progressPercentage);
                        //float lossValue = lossBuffer.get(0);

                        // Record the last loss.
                        if (batchIdx == 10 - 1) losses[epoch] = lossBuffer.get(0);
                    }

                    // Print the loss output for every 10 epochs.
                    if ((epoch + 1) % 10 == 0) {
                        System.out.println(
                                "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch]);
                    }
                }
                saveModelWeights(anotherInterpreter);
            } catch (IOException e){
                Log.e("ReportActivity","Error",e);
            }
            return null;
        }

        protected void onProgressUpdate(Integer... progress){
            super.onProgressUpdate(progress);
            progressBar.setProgress(progress[0]);
        }

        protected void onPostExecute(Void result){
            super.onPostExecute(result);
            progressBar.setVisibility(View.GONE);
            text.setText("Training Completed");
            btConfirm.setVisibility(View.VISIBLE);
        }
    }
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //OnDeviceTraining.onDeviceTraining();
        setContentView(R.layout.activity_federatedlearning);
        text = findViewById(R.id.federated_learning_text);
        status = findViewById(R.id.federated_learning_status);
        btConfirm = findViewById(R.id.btConfirmFederatedLearning);
        status.setText("Learning From: OPPO R11\nML Objective: "+ MainActivity.string +"\nML Model Size: 0.82MB");
        btConfirm.setVisibility(View.GONE);

        btConfirm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), ReportActivity2.class);
                startActivity(intent);
            }
        });

        new TrainModelTask().execute();

        /*context = this;

        try (Interpreter anotherInterpreter = new Interpreter(loadModelFile(context.getAssets(),"model.tflite"))) {
            int NUM_EPOCHS = 100;
            int BATCH_SIZE = 100;
            int IMG_HEIGHT = 28;
            int IMG_WIDTH = 28;
            int NUM_TRAININGS = 60000;
            int NUM_BATCHES = NUM_TRAININGS / BATCH_SIZE;

            List<FloatBuffer> trainImageBatches = new ArrayList<>(NUM_BATCHES);
            List<FloatBuffer> trainLabelBatches = new ArrayList<>(NUM_BATCHES);

            // Prepare training batches.
            for (int i = 0; i < NUM_BATCHES; ++i) {
                ByteBuffer trainImageBuffer = ByteBuffer.allocateDirect(4 * IMG_HEIGHT * IMG_WIDTH).order(ByteOrder.nativeOrder());
                FloatBuffer trainImages = trainImageBuffer.asFloatBuffer();

                ByteBuffer trainLabelsBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                FloatBuffer trainLabels = trainLabelsBuffer.asFloatBuffer();

                // Fill the data values...
                trainImageBatches.add((FloatBuffer) trainImages.rewind());
                trainLabelBatches.add((FloatBuffer) trainLabels.rewind());
            }

            // Run training for a few steps.
            float[] losses = new float[NUM_EPOCHS];
            for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
                for (int batchIdx = 0; batchIdx < NUM_BATCHES; ++batchIdx) {
                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("x", trainImageBatches.get(batchIdx));
                    inputs.put("y", trainLabelBatches.get(batchIdx));

                    Map<String, Object> outputs = new HashMap<>();
                    FloatBuffer lossBuffer = FloatBuffer.allocate(1);
                    outputs.put("loss", lossBuffer);

                    anotherInterpreter.runSignature(inputs, outputs, "train");

                    //float lossValue = lossBuffer.get(0);

                    // Record the last loss.
                    if (batchIdx == NUM_BATCHES - 1) losses[epoch] = lossBuffer.get(0);
                }

                // Print the loss output for every 10 epochs.
                if ((epoch + 1) % 10 == 0) {
                    System.out.println(
                            "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch]);
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }*/
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public FloatBuffer readImageAsFloatBuffer(Context context, String assetPath, int imgWidth, int imgHeight) {
        try {
            InputStream is = context.getAssets().open(assetPath);
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imgHeight * imgWidth).order(ByteOrder.nativeOrder());
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            int[] pixels = new int[imgWidth * imgHeight];
            resizedBitmap.getPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight);

            for (int pixel : pixels) {
                float red = ((pixel >> 16) & 0xFF) / 255.0f;
                float green = ((pixel >> 8) & 0xFF) / 255.0f;
                float blue = (pixel & 0xFF) / 255.0f;
                floatBuffer.put(red);  // Just an example; adjust as needed for your model
            }
            floatBuffer.rewind();
            return floatBuffer;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public FloatBuffer readLabelAsFloatBuffer(Context context, String assetPath, int numClasses) {
        try {
            InputStream is = context.getAssets().open(assetPath);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            int labelIndex = Integer.parseInt(reader.readLine().trim());

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * numClasses).order(ByteOrder.nativeOrder());
            FloatBuffer labelBuffer = byteBuffer.asFloatBuffer();
            labelBuffer.put(labelIndex, 1.0f);  // One-hot encoding
            labelBuffer.rewind();
            return labelBuffer;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private void saveModelWeights(Interpreter interpreter) {
        String filename = "trained_model_weights.ckpt";
        File file = new File(context.getFilesDir(), filename);
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", file.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();

        interpreter.runSignature(inputs, outputs, "save");
        Log.d("FederatedLearning", "Model weights saved to " + file.getAbsolutePath());
    }
}
