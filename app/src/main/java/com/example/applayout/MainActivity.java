package com.example.applayout;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.example.applayout.Metrics.DeviceUsageActivity;
import com.example.applayout.Report.ReportActivity;

public class MainActivity extends BaseActivity {
    TextView status;
    Button btConfirm;
    Button btDeviceUsage;
    ListView lvApplications;
    String[] applications;
    public static String string;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setContentView(R.layout.activity_main);
        super.onCreate(savedInstanceState);
        lvApplications = findViewById(R.id.lvApplication);
        btConfirm = findViewById(R.id.btConfirmApplication);
        btDeviceUsage = findViewById(R.id.btDeviceUsage);
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

        btDeviceUsage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), DeviceUsageActivity.class);
                startActivity(intent);
            }
        });
    }
}
