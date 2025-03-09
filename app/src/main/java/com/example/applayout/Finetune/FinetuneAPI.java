package com.example.applayout.Finetune;

import java.util.concurrent.ExecutionException;

import android.util.Log;


/**
 * Exposes the public APIs of finetuning to client code.
 */
public class FinetuneAPI {

    /**
     * This is a blocking method as .get() is required to ensure knowledge of the doInBackground() method's completion.
     * To avoid blocking the UI thread, ensure that this function's caller is a separate Thread.
     */
    public static void finetune(String modelFileAbsolutePath, String datasetDirAbsolutePath, int numEpochs, int imgHeight, int imgWidth) {
        Log.d("FinetuneAPI", "In FinetuneAPI.finetune()");
        try {
            new FinetuneTask(modelFileAbsolutePath, datasetDirAbsolutePath, numEpochs, imgHeight, imgWidth).execute().get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

}
