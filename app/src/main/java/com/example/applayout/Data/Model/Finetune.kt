package com.example.applayout.Data.Model


data class Finetune(
    var num_epochs: Int = 0,
    var batch_size: Int = 0,
    var performance_json: String
)

