package com.example.applayout.utils;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

public class BoundedRunnableExecutorTest {
    private BoundedRunnableExecutor executor;

    @Before
    public void setUp() {
        executor = new BoundedRunnableExecutor(4, 10);
    }

    @After
    public void tearDown() {
        executor.shutdown();
    }

    @Test
    public void testExecutorProcessesTasks() throws InterruptedException {
        int numTasks = 5; // Number of tasks to submit
        AtomicInteger completedTaskCount = new AtomicInteger(0); // To count completed tasks
        CountDownLatch latch = new CountDownLatch(numTasks); // To wait for all tasks to complete

        // Submit tasks to the executor
        for (int i = 0; i < numTasks; i++) {
            executor.submit(() -> {
                try {
                    Thread.sleep(500); // Simulate work
                    completedTaskCount.incrementAndGet(); // Increment completed task count
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown(); // Signal task completion
                }
            });
        }

        // Wait for all tasks to complete
        latch.await();

        // Verify that all tasks were completed
        assertEquals("Not all tasks were completed", numTasks, completedTaskCount.get());
    }
}
