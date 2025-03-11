package com.example.applayout.Finetune;

import java.io.File;
import java.util.concurrent.ExecutionException;

import android.content.Context;
import android.util.Log;


/**
 * Exposes the public APIs of finetuning to client code.
 */
public class FinetuneAPI {

    /**
     * This is a blocking method as .get() is required to ensure knowledge of the doInBackground() method's completion.
     * To avoid blocking the UI thread, ensure that this function's caller is a separate Thread.
     */
    public static void finetune(Context context, File filesDir, String modelFileName, String datasetDirName, int numEpochs, int batchSize) {
        Log.d("FinetuneAPI", "In FinetuneAPI.finetune()");
        try {
            new FinetuneTask(context, filesDir, modelFileName, datasetDirName, numEpochs, batchSize).execute().get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

}
