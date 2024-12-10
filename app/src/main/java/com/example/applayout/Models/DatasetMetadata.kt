package com.example.applayout.Models

data class DatasetMetadata(
    val taskDescription: String? = "No description provided",
    val lastModified: Long? = System.currentTimeMillis(),
    val modelTask: String? = "No model task provided",
    val publicLink: String? = "",
    val classLabels: List<String>? = emptyList()
)
