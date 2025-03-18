package com.example.applayout.utils

import android.content.Context
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.platform.LocalContext
import java.io.File
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.patrykandpatrick.vico.compose.chart.Chart
import com.patrykandpatrick.vico.compose.chart.line.lineChart
import com.patrykandpatrick.vico.core.entry.ChartEntry
import com.patrykandpatrick.vico.core.entry.ChartEntryModelProducer

fun readGraphData(context: Context, fileName: String): List<Pair<Float, Float>> {
    val file = File(context.filesDir, fileName)
    if (!file.exists()) return emptyList()

    return file.readLines().drop(1).mapNotNull { line ->
        val parts = line.split(",")
        if (parts.size == 2) parts[0].toFloat() to parts[1].toFloat() else null
    }
}

fun toPairs(list1: List<Float>, list2: List<Float>): List<Pair<Float, Float>> {
    require(list1.size == list2.size) { "Both lists must have the same length, but got ${list1.size} and ${list2.size}." }
    return list1.mapIndexed { index, value -> value to list2[index] }
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

@Composable
fun Graph(
    fileName: String? = null,
    xAxisTitle: String,
    yAxisTitle: String,
    useCase: String,
    xAxisData: List<Float>? = null,
    yAxisData: List<Float>? = null
) {
    val context = LocalContext.current
    val tag = "GraphUtil"

    var graphData: List<Pair<Float, Float>> = emptyList()
    if (fileName != null) {
        graphData = remember { readGraphData(context, fileName) }
    } else if (xAxisData != null && yAxisData != null) {
        Log.d(tag, "xAxisData: $xAxisData for useCase: $useCase")
        Log.d(tag, "yAxisData: $yAxisData for useCase: $useCase")
        graphData = remember { toPairs(xAxisData, yAxisData) }
    }
    Log.d(tag, "graphData: $graphData for useCase: $useCase")

    val chartEntries = remember { graphData.map { (epoch, loss) -> CustomChartEntry(epoch, loss) } }
    Log.d(tag, "chartEntries: $chartEntries for useCase: $useCase")
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
                "Graph of $yAxisTitle against number of $xAxisTitle for $useCase",
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp
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