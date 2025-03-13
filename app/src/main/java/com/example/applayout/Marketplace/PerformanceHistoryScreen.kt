package com.example.applayout.Marketplace

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.text.font.FontWeight
import com.example.applayout.Data.Model.Finetune
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

@Composable
fun PerformanceHistoryScreen(list: List<Finetune>) {
    Column(modifier = Modifier.padding(16.dp)) {
        Text(
            text = "Model Training Performance History",
            fontWeight = FontWeight.Bold,
            fontSize = 20.sp,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // Loop through each Finetune item and display its details
        list.forEach { item ->

            val performanceJsonMap: HashMap<String, Any> = Gson().fromJson(
                item.performance_json,
                object : TypeToken<HashMap<String, Any>>() {}.type
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Epochs: ${item.num_epochs}", fontWeight = FontWeight.Bold)
                    Text("Batch Size: ${item.batch_size}", fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))

                    Text("Performance Details:", fontWeight = FontWeight.Bold)
                    performanceJsonMap.forEach { (key, value) ->
                        Text("$key: $value")
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    // Dataset ID
                    Text("Dataset ID: ${item.dataset}", fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
