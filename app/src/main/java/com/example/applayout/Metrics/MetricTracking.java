package com.example.applayout.Metrics;

import android.os.Handler;
import android.os.HandlerThread;

import java.util.HashMap;

public class MetricTracking {

    /**
     * Performs a task and tracks the device usage during the task's execution.
     *
     * @param callback: A Runnable representing the task / function. If your task requires arguments, wrap the function call in a lambda.
     * @return A HashMap with the keys: [minCpu, avgCpu, maxCpu, minMem, avgMem, maxMem]
     */
    public static HashMap<String, Object> doWithTracking(Runnable callback) {
        return new MetricTracker().doWithTracking(callback);
    }

    private static class MetricTracker {
        private final Handler handler;
        private final int DELAY_MILLIS = 1000;
        private double minCpu = Double.MAX_VALUE;
        private double maxCpu = Double.MIN_VALUE;
        private double totalCpu = 0.0;
        private int cpuSamples = 0;

        private double minMem = Long.MAX_VALUE;
        private double maxMem = Long.MIN_VALUE;
        private double totalMem = 0;
        private int memSamples = 0;

        private final HashMap<String, Object> trackingResults = new HashMap<>();

        private final HandlerThread handlerThread;

        private MetricTracker() {
            handlerThread = new HandlerThread("MetricTrackingThread");
            handlerThread.start();
            handler = new Handler(handlerThread.getLooper());  // Create handler for background thread
            System.out.println("MetricTracking instance created.");
        }

        private HashMap<String, Object> doWithTracking(Runnable callback) {
            System.out.println("Starting Tracking...");
            startTracking();
            callback.run();
            stopTracking();
            System.out.println("Stopping Tracking...");

            return trackingResults;
        }

        private void startTracking() {
            handler.postDelayed(trackingRunnable, DELAY_MILLIS);
        }

        private void stopTracking() {
            handler.removeCallbacks(trackingRunnable);

            // Calculate averages
            double avgCpu = cpuSamples > 0 ? totalCpu / cpuSamples : 0.0;
            double avgMem = memSamples > 0 ? totalMem / memSamples : 0.0;

            // Save results in the HashMap
            trackingResults.put("minCpu", minCpu);
            trackingResults.put("maxCpu", maxCpu);
            trackingResults.put("avgCpu", avgCpu);

            trackingResults.put("minMem", minMem);
            trackingResults.put("maxMem", maxMem);
            trackingResults.put("avgMem", avgMem);

            // Stop the handler thread
            handlerThread.quitSafely();
        }

        private final Runnable trackingRunnable = new Runnable() {
            @Override
            public void run() {
                double[] usage = MetricTrackingUtils.calculateAppCpuAndMemUsage();
                double currentCpu = usage[0];
                double currentMem = usage[1];

                // Update CPU usage
                minCpu = Math.min(minCpu, currentCpu);
                maxCpu = Math.max(maxCpu, currentCpu);
                totalCpu += currentCpu;
                cpuSamples++;

                // Update memory usage
                minMem = Math.min(minMem, currentMem);
                maxMem = Math.max(maxMem, currentMem);
                totalMem += currentMem;
                memSamples++;

                // Continue tracking
                handler.postDelayed(this, DELAY_MILLIS);
            }
        };
    }

}
