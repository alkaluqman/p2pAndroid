package com.example.applayout.Finetune;

import java.util.concurrent.ExecutionException;

/**
 * Exposes the public APIs of finetuning to client code.
 */
public class FinetuneAPI {

    /**
     * This is a blocking method as .get() is required to ensure knowledge of the doInBackground() method's completion.
     * To avoid blocking the UI thread, ensure that this function's caller is a separate Thread.
     */
    private static void finetune(String modelFileAbsolutePath, String datasetDirAbsolutePath, int numEpochs, int batchSize, int imgHeight, int imgWidth, int numTrainings) {
        try {
            new FinetuneTask(modelFileAbsolutePath, datasetDirAbsolutePath, numEpochs, batchSize, imgHeight, imgWidth, numTrainings).execute().get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

}
