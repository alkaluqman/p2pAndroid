package com.example.applayout.Data.Model


data class Finetune(
    var train_test_split: Long = 0,
    var learning_rate: Long = 0,
    var epoch_number: Int = 0,
    var optimizer: String = "",
    var batch_size: Int = 0,
    var regularization: String = "",
    var dropout_rate: Long = 0
)

