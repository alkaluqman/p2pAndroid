package com.example.applayout;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.example.applayout.Finetune.FederatedLearningActivity;
import com.example.applayout.Report.ReportActivity;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class MainActivity extends BaseActivity {
    TextView status;
    Button btConfirm;
    Button btTrain;
    ListView lvApplications;
    String[] applications;
    public static String string;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setContentView(R.layout.activity_main);
        super.onCreate(savedInstanceState);
        lvApplications = findViewById(R.id.lvApplication);
        btConfirm = findViewById(R.id.btConfirmApplication);
        btTrain = findViewById(R.id.btTrain);
        status = findViewById(R.id.statusApplication);

        applications = new String[2];
        applications[0] = "Image Segmentation";
        applications[1] = "Object Detection";
        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, applications);
        lvApplications.setAdapter(arrayAdapter);

        lvApplications.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                status.setText("The Selected Application is " + applications[i]);
                string = applications[i];
            }
        });

        btConfirm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), ReportActivity.class);
                startActivity(intent);
            }
        });

        btTrain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                final boolean PROCESS_TEST = false;

                if (PROCESS_TEST) {
                    final int NUM_MODELS = 1;

                    final int APP_PID = android.os.Process.myPid();
                    System.out.println("APP_PID: " + APP_PID);

//                int currentUid = android.os.Process.myUid();
//                UserManager userManager = (UserManager) getSystemService(Context.USER_SERVICE);
//                int currentUserId = userManager.getUserHandle();
//                try {
//                    ProcessBuilder pb = new ProcessBuilder("whoami");
//                    Process userP = pb.start();
//                    BufferedReader r = new BufferedReader(new InputStreamReader(userP.getInputStream()));
//                    String l;
//                    while ((l = r.readLine()) != null) {
//                        System.out.println("Output: " + l);
//                    }
////                    System.out.println("User UID: " + currentUid);
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }

//                 EXPERIMENT USING PROCESSES
                    for (int i = 0; i < NUM_MODELS; i++) {
                        new Thread(() -> {
                            try {
                                ProcessBuilder processBuilder = new ProcessBuilder(
                                        "sh", "-c", "am start -n com.example.applayout.FederatedLearning/.FederatedLearningActivity");

                                Process process = processBuilder.start();

                                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                                String line;
                                while ((line = reader.readLine()) != null) {
                                    System.out.println("Output: " + line);
                                }

                                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                                while ((line = errorReader.readLine()) != null) {
                                    System.err.println("Error: " + line);
                                }

                                process.waitFor();

                            } catch (IOException | InterruptedException
                                    e) {
                                e.printStackTrace();
                            }
                        }).start();
                    }
                } else {
                    Intent intent = new Intent(view.getContext(), FederatedLearningActivity.class);
                    startActivity(intent);
                }
            }
        });


    }
}
