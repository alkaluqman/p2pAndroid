package com.example.applayout.Metrics;

import android.os.Process;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MetricTrackingUtils {

    private final static int APP_PID = Process.myPid();

    public static double[] calculateAppCpuAndMemUsage() {
        double totalCpu = 0.0;
        double totalMem = 0.0;
        try {
            // Fetch CPU usage for the parent process
            double[] usage = getCpuAndMemUsageForProcess(APP_PID);
            totalCpu += usage[0];
            totalMem += usage[1];

            // Fetch CPU usage for all child processes
            List<Integer> childPids = getChildProcesses();
            for (int pid : childPids) {
                usage = getCpuAndMemUsageForProcess(pid);
                totalCpu += usage[0];
                totalMem += usage[1];
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new double[]{totalCpu, totalMem};
    }

    public static List<Integer> getChildProcesses() {
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
            System.out.println("Child PIDs: ");
            System.out.println(childPids);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return childPids;
    }

    // Example Output of `top` command.
    // PID   | USER    | PR | NI  | VIRT | RES  | SHR  | S | [%CPU] | %MEM | TIME+   | ARGS
    // 31767 | u0_a483 | 10 | -10 |  16G | 232M | 142M | S |   0.0  | 3.1  | 0:37.45 | com.example.applayout
    //   0        1      2     3      4      5     6     7      8       9      10    |  11

    // Returns CPU, Mem
    public static double[] getCpuAndMemUsageForProcess(int pid) {
        try {
            String line;

            java.lang.Process p = Runtime.getRuntime().exec("top -b -n 1 -p " + pid);
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            while ((line = br.readLine()) != null) {
                if (line.contains(Integer.toString(pid))) {
                    String[] info = line.trim().replaceAll(" +", " ").split(" ");
                    if (info.length < 10) return new double[]{0.0, 0.0};
                    br.close();
                    return new double[]{Double.parseDouble(info[8]), Double.parseDouble(info[9])};
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new double[]{0.0, 0.0};
    }

}
