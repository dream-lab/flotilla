Fed Server
Fed Server gRPC sending models 
    
    fedserver_gRPC.send_model.init: empty_response
    fedserver_gRPC.send_model.cache_miss: client_id, status response 
    fedserver_gRPC.send_model.cache_hit: string-response,client_id,model
    fedserver_gRPC.send_model.timeout: client_id
    fedserver_gRPC.send_model.invaid.path: string_repsponse,path
    fedserver_gRPC.send_model.invalid_channel: client_id
    fedserver_gRPC.send_model.client.finished: string_response,client_id,time_taken
    fedserver_gRPC.send_model.finished: string_response,time_taken
    
    fedserver.init_client_model_sent: empty_response

Fed Server gRPC Bench
    
    fedserver_gRPC.bench.init: empty_response
    fedserver_gRPC.bench.connect: client_id
    fedserver_gRPC.bench.await.response: client_id
    fedserver_gRPC.bench.connection_terminated: client_id
    fedserver_gRPC.bench.client_dropped: client_id
    fedserver_gRPC.bench.invalid_channel: client_id
    fedserver_gRPC.bench.results: string_response,client_id,bench_time,num_mini_batches
    fedserver_gRPC.bench.client.finished: string_response,client_id,time
    fedserver_gRPC.bench.finished: string_response,time

Fed Server gRPC Training
    
    fedserver_gRPC.train.init: empty_response
    fedserver_gRPC.train.config: string_response,model-training_rounds-epochs-batch_size-loss_fun-optimizer-lr-timeout
    fedserver_gRPC.train.session.config: string_response,session_id-aggregator-client_selection and percentage
    fedserver_gRPC.train.connect: string_response,client_id
    fedserver_gRPC.train.await.response: client_id
    fedserver_gRPC.train.connection_terminated: client_id
    fedserver_gRPC.train.invalid_channel: client_id
    fedserver_gRPC.train.client_dropped: client_id
    
    fedserver_gRPC.train.rounds: training_rounds
    fedserver_gRPC.train.round.init: string_response, round number, no.of clients, selected clients sperated by,
    fedserver_gRPC.train.round.model_weights.pickle.time: string_response, round number, time
    fedserver_gRPC.train.round.gather.init: string_response, round number, time
    fedserver_gRPC.train.round.gather.finished : string_response, round number, time
    fedserver_gRPC.train.round.aggregation.init: string_response, round number, number of client weights aggregated
    fedserver_gRPC.train.round.aggregation.finish: string_response, round number, time
    fedserver_gRPC.train.round.validation.init: string_response, round number
    fedserver_gRPC.train.round.validation.finish: string_response, round number, time
    fedserver_gRPC.train.round.results: string_response,Client_id,round_no,epoch_count,training_time,training_loss,accuracy
    fedserver_gRPC.train.round.client.finished: string_response,client_id,round_no,time_taken
    fedserver_gRPC.train.round.aggregation.finish: string_response, time_taken
    fedserver_gRPC.train.round.validation.init: string_response, start_time
    fedserver_gRPC.train.round.validation.finish: string_response, time_taken
    fedserver_gRPC.train.round.finished: string_response,round number,run_time
    fedserver_gRPC.train.global.model.accuracy: string_response, round_no, accuracy
    fedserver_gRPC.train.global.model.loss: string_response, round_no, loss
    fedserver_gRPC.train.globalaccuracy: global accuracies for round1, round2, round3, ...
    fedserver_gRPC.train.globalloss: global loss for round1, round2, round3, ...

    fedserver_gRPC.train.finished: string_response,time_taken

server MQTT
    
    MQTT.server.init: empty_response
    MQTT.server.connect.request: connection_status
    MQTT.server.subscribe.request: request_code
    MQTT.server.publish.request: request_code
    MQTT.server.heartbeat.received: string_response, client_id
    MQTT.server.heartbeat.received.message: heartbeat_info
    MQTT.server.heartbeat.invalid.client: string_response, client_id
    MQTT.server.broker.connect: string_response, mqtt_broker, mqtt_port
    MQTT.server.subscribed.topics: string_response, mqtt_client_topic, heartbeat
    MQTT.server.advert.publish.topic: string_response, mqtt_server_topic
    MQTT.server.advert.publish.payload: string_response, client_id, payload
    MQTT.server.await.timeout: string_response
    MQTT.server.clients: string_response, client_payload

Fed Server client selection
    
    fedserver.client_selection.invalid.module: string_response, module_name

Fed Server Aggregator

    fedserver.aggregator.endof.weights: string_response

Fed Server Local Loss

    fedserver_gRPC.local_loss.round.results: client_id, round_no, loss
    fedserver_gRPC.local_loss.connection_terminated: client_id
    fedserver_gRPC.local_loss.invalid_channel: client_id
    fedserver_gRPC.local_loss.client_dropped: client_id

Fed Server Echo
    
    fedserver_gRPC.echo.init: string_response,no.of clients
    fedserver_gRPC.echo.start: string_response,client_id
    fedserver_gRPC.echo.invalid_channel: client_id
    fedserver_gRPC.echo.timeout: string_response,client_id
    fedserver_gRPC.echo.response: string_response,client_id
    fedserver_gRPC.echo.client.finished: string_response,client_id,time_taken
    fedserver_gRPC.echo.finished: string_response,time_taken

Fed server run

    fedserver_gRPC.echo.response.clients: string_response,no_of_clients,client_id1,client_id2,...

    
Fed server Keyboard Interrupt

    fedserver.keyboard_interrupt: string_response

Fed Client

client MQTT

    MQTT.client.init: empty_response
    MQTT.client.connect: response_code
    
    MQTT.client.subscribe: response_code
    MQTT.client.publish: response_code
    MQTT.client.advertise.response: advert_response_info  
    MQTT.client.broker.connect: string_response,mqtt_broker,mqtt_port
    MQTT.client.subscribed: string_response, topic
    MQTT.client.advert: string_response, topic
    MQTT.client.advert.payload: string_response, client_id , payload

    MQTT.client.heartbeat.payload: string_response, client_id, timestamp

Fed Client gRPC init

    fedclient_gRPC.init: empty_response

Fed Client Init

    fedclient.init: client_id

Fed Client Keyboard interrupt

    fedclient.Keyboard_interrupt: string_response
    fedclient.Keyboard_interrupt.exit : empty_response

Fed Client gRPC echo

    fedclient.gRPC.echo.request: string_response, received_msg

Fed Client gRPC model download
    
    fedclient.gRPC.download.model.init: empty_response
    fedclient.gRPC.download.model.exists: string_response

Fed Client gRPC Benchmark

    fedclient.gRPC.benchmark.init: empty_response

	fedclient.gRPC.benchmark.model.classes: string_response, classes_present
	fedclient.gRPC.benchmark.model.architecture: model_architecture
	fedclient.gRPC.benchmark.results: string_response, Number_of_mini_batches, time_taken, loss, accuracy
	fedclient.gRPC.benchmark.exception: exception

    fedclient.gRPC.benchmark.finish: string_response

Fed Client gRPC Training

    fedclient.gRPC.training.round.init: empty_response

	fedclient.gRPC.training.round.model: model_id
	fedclient.gRPC.training.round.weights: model_weight
	fedclient.gRPC.training.round.classes: string_response, classes_present
	fedclient.gRPC.training.round.results: string_response, Number_of_epochs, time_taken, loss, accuracy
	fedclient.gRPC.training.round.model.notfound: string_response

    fedclient.gRPC.training.round.complete: empty_response

Fed Client gRPC LocalLoss

    fedclient.gRPC.local_loss.round.init: empyty_response
    fedclient.gRPC.local_loss.round.model: model_id
    fedclient.gRPC.local_loss.round.model.notfound: string_response
    fedclient.gRPC.local_loss.round.complete: empty_response
    fedclient.gRPC.local_loss.round.results: client_id, round_no, loss
