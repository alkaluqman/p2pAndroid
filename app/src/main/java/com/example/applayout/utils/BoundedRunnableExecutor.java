package com.example.applayout.utils;

import java.util.concurrent.*;

public class BoundedRunnableExecutor {
    private final ThreadPoolExecutor executorService;

    public BoundedRunnableExecutor(int threadCapacity, int queueCapacity) {
        // Use a bounded BlockingQueue directly in the ThreadPoolExecutor
        this.executorService = new ThreadPoolExecutor(
                threadCapacity,            // Core pool size
                threadCapacity,            // Maximum pool size (fixed thread pool)
                0L,                  // Keep-alive time (not used here)
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(queueCapacity), // Bounded task queue
                new ThreadPoolExecutor.CallerRunsPolicy() // Rejection policy
        );
    }

    public void submit(Runnable task) {
        executorService.execute(task); // Submit directly to the executor
    }

    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(30, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

//    public static void main(String[] args) throws InterruptedException {
//        int capacity = 3;         // Maximum concurrent tasks
//        int queueCapacity = 5;    // Maximum waiting tasks in the queue
//        BoundedRunnableExecutor executor = new BoundedRunnableExecutor(capacity, queueCapacity);
//
//        // Example tasks
//        for (int i = 0; i < 10; i++) {
//            int taskId = i;
//            executor.submit(() -> {
//                System.out.println("Running task " + taskId + " on thread " + Thread.currentThread().getName());
//                try {
//                    Thread.sleep(1000); // Simulate work
//                } catch (InterruptedException e) {
//                    Thread.currentThread().interrupt();
//                }
//            });
//        }
//
//        // Shutdown after all tasks complete
//        executor.shutdown();
//    }
}
