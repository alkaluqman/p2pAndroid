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
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import android.net.TrafficStats;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import com.example.applayout.R;

public class DeviceUsageActivity extends AppCompatActivity {
    private ProgressBar availableMemoryProgressBar;
    private ProgressBar appUsedMemoryProgressBar;
    private TextView memoryUsageText;
    private TextView cpuUsageText;
    private TextView networkUsageText;
    private TextView batteryUsageText;
    private Handler handler;
    private long lastRxBytes = 0;
    private long lastTxBytes = 0;

    private final int APP_PID = Process.myPid();

    private final int DELAY_MILLIS = 1000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_device_usage);

        availableMemoryProgressBar = findViewById(R.id.availableMemoryProgressBar);
        availableMemoryProgressBar.setVisibility(View.VISIBLE);

        appUsedMemoryProgressBar = findViewById(R.id.appUsedMemoryProgressBar);
        appUsedMemoryProgressBar.setVisibility(View.VISIBLE);

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
                handler.postDelayed(this, DELAY_MILLIS); // Update every second
            }
        }, DELAY_MILLIS);
    }

    private void displayResourceUsage() {
        displayMemoryUsage();
        displayCpuUsage();
        displayNetworkUsage();
        displayBatteryUsage();
        getChildProcesses();
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

        long totalMemory = memoryInfo.totalMem / (1024 * 1024);

        Runtime runtime = Runtime.getRuntime();
        long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
        long maxMemory = runtime.maxMemory() / (1024 * 1024);

        double availableMemoryPercentage = (availableMemory * 100.0) / totalMemory;
        availableMemoryProgressBar.setProgress((int) availableMemoryPercentage);

        double appUsedMemoryPercentage = (usedMemory * 100.0) / maxMemory;
        appUsedMemoryProgressBar.setProgress((int) appUsedMemoryPercentage);

        String memoryUsageStr = "Device Available Memory: " + availableMemory + " MB\n" +
                "Device Total Memory: " + totalMemory + " MB\n" +
                "App Used Memory: " + usedMemory + " MB\n" +
                "App Max Memory: " + maxMemory + " MB";

        memoryUsageText.setText(memoryUsageStr);
    }

    private void displayCpuUsage() {
        double cpuUsage = calculateAppCpuUsage();
        String cpuUsageStr = "App CPU Usage: " + String.format("%.2f", cpuUsage) + "%";
        cpuUsageText.setText(cpuUsageStr);
    }

    private double calculateAppCpuUsage() {
        double totalCpu = 0.0;
        try {
            // Fetch CPU usage for the parent process
            totalCpu += getCpuUsageForProcess(APP_PID);

            // Fetch CPU usage for all child processes
            List<Integer> childPids = getChildProcesses();
            for (int pid : childPids) {
                totalCpu += getCpuUsageForProcess(pid);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return totalCpu;
    }

    private List<Integer> getChildProcesses() {
        List<Integer> childPids = new ArrayList<>();
        try {
            String command = "ps -o pid,ppid | grep " + APP_PID;

            java.lang.Process process = Runtime.getRuntime().exec(new String[]{"sh", "-c", command});
            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;

            while ((line = br.readLine()) != null) {
                line = line.trim();
                String[] parts = line.split("\\s+");
                if (parts.length >= 2) {
                    int ppid = Integer.parseInt(parts[1]);
                    int pid = Integer.parseInt(parts[0]);
                    if (ppid == APP_PID) {
                        childPids.add(pid);
                    }
                }
            }
            br.close();
//            System.out.println("Child PIDs: ");
//            System.out.println(childPids);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return childPids;
    }

    // Example Output of `top` command.
    // PID   | USER    | PR | NI  | VIRT | RES  | SHR  | S | [%CPU] | %MEM | TIME+   | ARGS
    // 31767 | u0_a483 | 10 | -10 |  16G | 232M | 142M | S |   0.0  | 3.1  | 0:37.45 | com.example.applayout
    //   0        1      2     3      4      5     6     7      8       9      10    |  11

    private double getCpuUsageForProcess(int pid) {
        try {
            String line;

            java.lang.Process p = Runtime.getRuntime().exec("top -b -n 1 -p " + pid);
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            while ((line = br.readLine()) != null) {
                if (line.contains(Integer.toString(pid))) {
                    String[] info = line.trim().replaceAll(" +", " ").split(" ");
                    if (info.length < 9) return 0.0;
                    br.close();
                    return Double.parseDouble(info[8]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0.0;
    }
}
