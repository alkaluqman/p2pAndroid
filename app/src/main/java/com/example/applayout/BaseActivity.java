package com.example.applayout;

import android.content.Intent;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import com.example.applayout.Metrics.DeviceUsageActivity;

public class BaseActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        FloatingActionButton fab = findViewById(R.id.deviceusage_fab);

        if (fab != null) {
            fab.setOnTouchListener(new View.OnTouchListener() {
                private static final int MOVE_THRESHOLD = 1500;  // Minimum distance to be considered a drag
                private float dX, dY;
                private boolean isDragging = false;
                private long touchDownTime;

                @Override
                public boolean onTouch(View v, MotionEvent event) {

                    switch (event.getAction()) {
                        case MotionEvent.ACTION_DOWN:
                            // Record the initial touch position and time
                            dX = v.getX() - event.getRawX();
                            dY = v.getY() - event.getRawY();
                            touchDownTime = System.currentTimeMillis();
                            isDragging = false;  // Reset dragging flag
                            return true;  // Consume the event
                        case MotionEvent.ACTION_MOVE:
                            // Check if the move is large enough to be considered a drag
                            if (!isDragging && (Math.abs(event.getRawX() - dX) > MOVE_THRESHOLD || Math.abs(event.getRawY() - dY) > MOVE_THRESHOLD)) {
                                isDragging = true;
                            }

                            // If it's a drag, move the FAB
                            if (isDragging) {
                                v.animate()
                                        .x(event.getRawX() + dX)
                                        .y(event.getRawY() + dY)
                                        .setDuration(0)
                                        .start();
                            }
                            return true;  // Consume the event
                        case MotionEvent.ACTION_UP:
                            // If the user tapped (no significant movement), handle the click
                            if (!isDragging) {
                                long touchUpTime = System.currentTimeMillis();
                                // If the touch duration is short (considered a click)
                                if (touchUpTime - touchDownTime < 300) {  // Adjust duration as needed
                                    // Trigger the click action here
                                    handleFabClick(v);
                                    v.performClick();
                                }
                                return true;  // Consume the event
                            }
                            isDragging = false;
                            return true;  // If it was a drag, do nothing
                        default:
                            return false;
                    }
                }
            });

        }
    }

    private void handleFabClick(View v) {
        Intent intent = new Intent(v.getContext(), DeviceUsageActivity.class);
        v.getContext().startActivity(intent);
    }
}

