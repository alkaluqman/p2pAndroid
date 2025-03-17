package com.example.applayout.Finetune;

import static com.example.applayout.Finetune.FinetuneUtils.extractWeights;
import static com.example.applayout.Finetune.FinetuneUtils.saveWeightsToCheckpoint;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An instance of this class represents an instance of Finetuning.
 */
public class FederatedLearnTask extends AsyncTask<Void, Integer, Void> {

    private final Context context;
    private final File filesDir;
    private final List<String> modelFileNames;
    private final String newModelFileName;

    public FederatedLearnTask(Context context, File filesDir, List<String> modelFileNames, String newModelFileName) {
        this.context = context.getApplicationContext();
        this.filesDir = filesDir;
        this.modelFileNames = modelFileNames;
        this.newModelFileName = newModelFileName;
    }

    protected void onPreExecute() {
        super.onPreExecute();
    }

    protected Void doInBackground(Void... voids) {

        federatedLearnManual(context, filesDir, modelFileNames, newModelFileName);
        return null;
    }

    protected void onProgressUpdate(Integer... progress) {
        super.onProgressUpdate(progress);
    }

    protected void onPostExecute(Void result) {
        super.onPostExecute(result);
    }

    private static FloatBuffer averageFloatBuffers(List<FloatBuffer> buffers) {
        if (buffers.isEmpty()) return null;
        int size = buffers.get(0).capacity();
        for (FloatBuffer buffer : buffers) {
            if (buffer.capacity() != size) {
                throw new IllegalArgumentException("All FloatBuffers must have the same size.");
            }
        }
        FloatBuffer averagedBuffer = FloatBuffer.allocate(size);
        float[] sumArray = new float[size];
        for (FloatBuffer buffer : buffers) {
            buffer.rewind();
            for (int i = 0; i < size; i++) {
                sumArray[i] += buffer.get();
            }
        }
        int numBuffers = buffers.size();
        for (int i = 0; i < size; i++) {
            averagedBuffer.put(sumArray[i] / numBuffers);
        }
        averagedBuffer.rewind();
        return averagedBuffer;
    }
    private void federatedLearnManual(Context context, File filesDir, List<String> modelFileNames, String newModelFileName) {
        Log.d(this.getClass().getName(), "Beginning federatedLearnManual...");
        final String MODEL_EXT = ".ckpt";

        try {
            List<Map<String, FloatBuffer>> allWeights = new ArrayList<>();

            for (String name : modelFileNames) {
                Interpreter anotherInterpreter = new Interpreter(FinetuneUtils.loadModelFile(context.getAssets(), "model.tflite"));
                String modelFileAbsolutePath = FinetuneUtils.getAbsolutePathFromFilesDir(filesDir, "models", name + MODEL_EXT);
                FinetuneUtils.restoreWeightsFromCheckpoint(anotherInterpreter, modelFileAbsolutePath);
                allWeights.add(extractWeights(anotherInterpreter));
            }
            Map<String, FloatBuffer> averagedWeights = new HashMap<>();
            for (String key : allWeights.get(0).keySet()) {
                List<FloatBuffer> buffersForKey = allWeights.stream()
                        .map(map -> map.get(key))
                        .collect(Collectors.toList());
                averagedWeights.put(key, averageFloatBuffers(buffersForKey));
            }
//            for (Map.Entry<String, FloatBuffer> entry : averagedWeights.entrySet()) {
//                    FloatBuffer buffer = (FloatBuffer) entry.getValue();
//                    buffer.rewind();
//                    StringBuilder sb = new StringBuilder();
//                    sb.append(entry.getKey()).append(": [");
//                    for (int i = 0; i < Math.min(5, buffer.capacity()); i++) {
//                        sb.append(buffer.get()).append(" ");
//                    }
//                    sb.append("...]");
//                    Log.d("federated_learn", sb.toString());
//                }
            saveWeightsToCheckpoint(filesDir, new Interpreter(FinetuneUtils.loadModelFile(context.getAssets(), "model.tflite")), averagedWeights, newModelFileName + MODEL_EXT);
            Log.d(this.getClass().getName(), "Completed federatedLearnManual!");
        } catch (IOException e) {
            Log.e(this.getClass().getName(), "Error", e);
        }
    }

}
