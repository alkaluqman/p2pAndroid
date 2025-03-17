package com.example.applayout.Finetune;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutionException;


/**
 * Exposes the public APIs of finetuning to client code.
 */
public class FinetuneAPI {

    /**
     * This is a blocking method as .get() is required to ensure knowledge of the doInBackground() method's completion.
     * To avoid blocking the UI thread, ensure that this function's caller is a separate Thread.
     */
    public static void finetune(Context context, File filesDir, String modelFileName, String datasetDirName, int numEpochs, int batchSize, String newModelFileName) {
        Log.d("FinetuneAPI", "In FinetuneAPI.finetune()");
        try {
            new FinetuneTask(context, filesDir, modelFileName, datasetDirName, numEpochs, batchSize, newModelFileName).execute().get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    public static void federatedLearn(Context context, File filesDir, List<String> modelFileNames, String newModelFileName) {
        Log.d("FinetuneAPI", "In FinetuneAPI.federatedLearn()");
        try {
            new FederatedLearnTask(context, filesDir, modelFileNames, newModelFileName).execute().get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    public static void getBaseCkptFile(Context context, File filesDir, String modelFileName) {
        FinetuneUtils.getBaseCkptFile(context, filesDir, modelFileName);
    }

}
