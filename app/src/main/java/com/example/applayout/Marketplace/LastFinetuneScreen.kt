package com.example.applayout.Marketplace

import android.util.Log
import android.content.Context
import java.io.File
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.patrykandpatrick.vico.compose.chart.Chart
import com.patrykandpatrick.vico.compose.chart.line.lineChart
import com.patrykandpatrick.vico.core.entry.ChartEntry
import com.patrykandpatrick.vico.core.entry.ChartEntryModelProducer


fun readLossData(context: Context, fileName: String): List<Pair<Float, Float>> {
    val file = File(context.filesDir, fileName)
    if (!file.exists()) return emptyList()

    return file.readLines().drop(1).mapNotNull { line ->
        val parts = line.split(",")
        if (parts.size == 2) parts[0].toFloat() to parts[1].toFloat() else null
    }
}

@Composable
fun LastFinetuneScreen() {
    val context = LocalContext.current
    val lossData = remember { readLossData(context, "last_run_losses.csv") }
    Log.d("LastFinetuneScreen", "lossData: $lossData")

    val chartEntries = remember { lossData.map { (epoch, loss) -> CustomChartEntry(epoch, loss) } }
    Log.d("LastFinetuneScreen", "chartEntries: $chartEntries")
    val chartEntryProducer = remember(chartEntries) {
        ChartEntryModelProducer(chartEntries)
    }
    val chartState = rememberUpdatedState(chartEntryProducer)


    if (chartEntries.isEmpty()) {
        Text("No data available", modifier = Modifier.padding(16.dp))
        return
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp) // Add vertical space between children
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            Text(
                "Graph of loss against number of epochs",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        }

        // Chart with vertical spacing before it starts
        Chart(
            chart = lineChart(),
            chartModelProducer = chartState.value,
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp)
        )
    }

}

// Custom class for chart entries
data class CustomChartEntry(
    override val x: Float,
    override val y: Float
) : ChartEntry {
    override fun withY(y: Float): ChartEntry {
        return CustomChartEntry(x, y)
    }
}
