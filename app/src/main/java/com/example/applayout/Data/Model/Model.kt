package com.example.applayout.Data.Model

data class Model(
    var uniqueIdentifier: String,
    var model_task: String,
    var last_trained: String,
    var description: String,
    var weight_size: Long,
    var is_uploaded: Boolean,
    var usage: Int,
    var likes: Int,
    var public_link: String,
    var architecture: String,
    var isOwner: Boolean,
)
