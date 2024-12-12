package com.example.applayout.Communications;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.applayout.LocalAssets.LocalAssetActivity;
import com.example.applayout.R;

public class CommunicationActivity extends AppCompatActivity {
    Button btWifi, btBluetooth, btMarketPlace;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_communications);
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
