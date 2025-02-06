package com.example.applayout.Communications;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;


import com.example.applayout.LocalAssets.LocalAssetActivity;
import com.example.applayout.R;

    
import com.example.applayout.BaseActivity;

public class CommunicationActivity extends BaseActivity {
    Button btWifi, btBluetooth, btMarketPlace;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setContentView(R.layout.activity_communications);
        super.onCreate(savedInstanceState);
        btWifi = findViewById(R.id.btWifi);
        btBluetooth = findViewById(R.id.btBluetooth);
        btMarketPlace= findViewById(R.id.btMarketplace);

        btWifi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), WifiActivity.class);
                startActivity(intent);
            }
        });

        btBluetooth.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), BluetoothActivity.class);
                startActivity(intent);
            }
        });

        btMarketPlace.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), LocalAssetActivity.class);
                startActivity(intent);
            }
        });
    }
}
