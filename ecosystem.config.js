module.exports = {
    apps: [{
        name: 'face-model',
        script: './index.js',
        watch: true,
        max_restarts: 5,
        restart_delay: 1000,
        out_file: "./out.log",
        error_file: "./error.log",
        merge_logs: true,
        log_date_format: "DD-MM HH:mm:ss Z",
        log_type: "json",
        PORT: 9857,
        env: {
            NODE_ENV: 'production'
        }
    }]
};