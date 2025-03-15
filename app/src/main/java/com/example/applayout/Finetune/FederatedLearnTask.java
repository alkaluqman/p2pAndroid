package com.example.applayout.Finetune;

import static com.example.applayout.Finetune.FinetuneUtils.averageTensorMaps;
import static com.example.applayout.Finetune.FinetuneUtils.extractWeights;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    private void federatedLearnManual(Context context, File filesDir, List<String> modelFileNames, String newModelFileName) {
        Log.d(this.getClass().getName(), "Beginning federatedLearnManual...");
        final String MODEL_EXT = ".ckpt";

        try {
            List<Map<String, Tensor>> allWeights = new ArrayList<>();

            for (String name : modelFileNames) {
                // Load the base model from assets
                Interpreter anotherInterpreter = new Interpreter(FinetuneUtils.loadModelFile(context.getAssets(), "model.tflite"));
                String modelFileAbsolutePath = FinetuneUtils.getAbsolutePathFromFilesDir(filesDir, "models", name + MODEL_EXT);
                FinetuneUtils.restoreWeightsFromCheckpoint(anotherInterpreter, modelFileAbsolutePath);
                allWeights.add(extractWeights(anotherInterpreter));
            }
            Map<String, TensorBuffer> combinedWeights = averageTensorMaps(allWeights);


            Log.d(this.getClass().getName(), "Completed federatedLearnManual!");
        } catch (IOException e) {
            Log.e(this.getClass().getName(), "Error", e);
        }
    }

}
