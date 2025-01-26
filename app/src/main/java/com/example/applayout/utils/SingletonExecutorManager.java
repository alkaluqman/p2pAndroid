package com.example.applayout.utils;

public class SingletonExecutorManager {
    private static final int THREAD_POOL_CAPACITY = 4; // Adjust capacity
    private static final int QUEUE_CAPACITY = 10;     // Adjust queue size

    private static BoundedRunnableExecutor instance;

    // Private constructor to prevent instantiation
    private SingletonExecutorManager() {
    }

    // Public method to get the singleton instance
    public static synchronized BoundedRunnableExecutor getInstance() {
        if (instance == null) {
            instance = new BoundedRunnableExecutor(THREAD_POOL_CAPACITY, QUEUE_CAPACITY);
        }
        return instance;
    }

    // Graceful shutdown
    public static synchronized void shutdown() {
        if (instance != null) {
            instance.shutdown();
        }
    }
}
