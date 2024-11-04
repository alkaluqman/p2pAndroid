package com.example.applayout.Metrics;

import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.text.format.Formatter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.net.TrafficStats;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import com.example.applayout.R;

public class DeviceUsageActivity extends AppCompatActivity {

    private TextView memoryUsageText;
    private TextView cpuUsageText;
    private TextView networkUsageText;
    private TextView batteryUsageText;
    private Handler handler;
    private long lastRxBytes = 0;
    private long lastTxBytes = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_device_usage);

        memoryUsageText = findViewById(R.id.memoryUsageText);
        cpuUsageText = findViewById(R.id.cpuUsageText);
        networkUsageText = findViewById(R.id.networkUsageText);
        batteryUsageText = findViewById(R.id.batteryUsageText);
        handler = new Handler(Looper.getMainLooper());

        lastRxBytes = TrafficStats.getUidRxBytes(Process.myUid());
        lastTxBytes = TrafficStats.getUidTxBytes(Process.myUid());

        startUsageMonitoring();
    }

    private void startUsageMonitoring() {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                displayResourceUsage();
                handler.postDelayed(this, 1000); // Update every second
            }
        }, 1000);
    }

    private void displayResourceUsage() {
        displayMemoryUsage();
        displayCpuUsage();
        displayNetworkUsage();
        displayBatteryUsage();
    }

    private void displayBatteryUsage() {
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = registerReceiver(null, ifilter);

        int level = batteryStatus != null ? batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) : -1;
        int scale = batteryStatus != null ? batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1) : -1;
        int batteryPct = (int) ((level / (float) scale) * 100);

        boolean isCharging = batteryStatus != null &&
                (batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1) == BatteryManager.BATTERY_STATUS_CHARGING ||
                        batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1) == BatteryManager.BATTERY_STATUS_FULL);

        String chargingStatus = isCharging ? "Charging" : "Not Charging";

        String batteryUsageStr = "Battery Level: " + batteryPct + "%\n" +
                "Battery Status: " + chargingStatus;

        batteryUsageText.setText(batteryUsageStr);
    }

    private void displayNetworkUsage() {
        long currentRxBytes = TrafficStats.getUidRxBytes(Process.myUid());
        long currentTxBytes = TrafficStats.getUidTxBytes(Process.myUid());
        long rxBytes = currentRxBytes - lastRxBytes;
        long txBytes = currentTxBytes - lastTxBytes;
        lastRxBytes = currentRxBytes;
        lastTxBytes = currentTxBytes;

        String rxBytesFormatted = Formatter.formatFileSize(this, rxBytes);
        String txBytesFormatted = Formatter.formatFileSize(this, txBytes);

        String networkUsageStr = "Network Download: " + rxBytesFormatted + "\n" +
                "Network Upload: " + txBytesFormatted;

        networkUsageText.setText(networkUsageStr);
    }

    private void displayMemoryUsage() {
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memoryInfo);

        long availableMemory = memoryInfo.availMem / (1024 * 1024);
        String memoryUsageStr = getMemoryUsageStr(memoryInfo, availableMemory);

        memoryUsageText.setText(memoryUsageStr);
    }

    @NonNull
    private static String getMemoryUsageStr(ActivityManager.MemoryInfo memoryInfo, long availableMemory) {
        long totalMemory = memoryInfo.totalMem / (1024 * 1024);

        Runtime runtime = Runtime.getRuntime();
        long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
        long maxMemory = runtime.maxMemory() / (1024 * 1024);

        return "Device Available Memory: " + availableMemory + " MB\n" +
                "Device Total Memory: " + totalMemory + " MB\n" +
                "App Used Memory: " + usedMemory + " MB\n" +
                "App Max Memory: " + maxMemory + " MB";
    }

    private void displayCpuUsage() {
        double cpuUsage = getAppCpuUsage();
        String cpuUsageStr = "App CPU Usage: " + String.format("%.2f", cpuUsage) + "%";
        cpuUsageText.setText(cpuUsageStr);
    }

    private double getAppCpuUsage() {
        try {
            String Result;
            String pid = Integer.toString(Process.myPid());

            java.lang.Process p = Runtime.getRuntime().exec("top -n 1");
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            while ((Result = br.readLine()) != null) {
                if (Result.contains(pid)) {
                    String[] info = Result.trim().replaceAll(" +", " ").split(" ");
                    return Double.valueOf(info[9]);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0.0;
    }
}
