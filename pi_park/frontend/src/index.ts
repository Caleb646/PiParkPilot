import { fabric } from 'fabric';

interface recv_data {
	aspect_ratio: number | null; // (z_end - z_start) / (x_end - x_start) or height / width
	cur_pos: [number, number, number] | null; // x, z, y rotation
	cur_path?: number[][]; // normalized x and y coords
}

window.onload = () => {
	let CANVAS_WIDTH = 640; 
	let CANVAS_HEIGHT = CANVAS_WIDTH; // * aspect_ratio;
	let aspect_ratio = 1; // 0.34433285509325684; // height / width
	
	const canvas = new fabric.StaticCanvas('canvas');
	canvas.setDimensions({width: CANVAS_WIDTH + 25, height: CANVAS_HEIGHT + 25 });

	//let prev_pos: number[] = [0, 0, 0, 0]; // x, z, y_rotation
	//let current_pos: number[] = [0, 0, 0, 0]; // x, z, y_rotation
	//let path = [[0.0, 0.0], [0.0012686859714564692, 2.1421676329816692e-05], [0.009117087530724217, 0.0010935504362897944], [0.027560712916798777, 0.009675567693824001], [0.058693304756706983, 0.04093591986167883], [0.10423550203349648, 0.11365513763522196], [0.16703720219807877, 0.23832877544243206], [0.25113285286742393, 0.4070113669798277], [0.35948000501656, 0.5929886330201721], [0.4901612511126707, 0.7616712245575679], [0.6334035773770547, 0.8863448623647783], [0.7719931333258233, 0.9590640801383215], [0.8861103038932463, 0.9903244323061761], [0.9610653406443918, 0.9989064495637104], [0.9945111356272253, 0.9999785783236705]];
	const Y_MAX = 1;

	function points_to_lines(points: number[][]) {
		const result = [];
		for(let i = 0; i < points.length - 1; ++i) {
			const prev = points[i];
			const cur = points[i+1];
			let line = new fabric.Line(
				[prev[0] * CANVAS_WIDTH, (Y_MAX - prev[1]) * CANVAS_HEIGHT, cur[0] * CANVAS_WIDTH, (Y_MAX - cur[1]) * CANVAS_HEIGHT],
				{
					strokeWidth: 3,
					fill: 'red',
					stroke: 'red',
					//scaleX: -1,
					//scaleY: -1,
					// originX: 'center',
					// originY: 'center'
				}
			);
			result.push(line);
		}
		return result;
	}
	const indicator = new fabric.Triangle({
	  left: 0, // x
	  top: 1 * CANVAS_HEIGHT, // z
	  angle: 90, // y rotation
	  width: 15,
	  height: 15,
	  fill: 'black',
	  originX: 'center',
	  originY: 'center'
	});
	canvas.add(indicator);

	function update_canvas(data: recv_data) {
		if(data.cur_pos != null) {
			indicator.set({
				//left: data.cur_pos[0] * CANVAS_WIDTH, // x
				left: data.cur_pos[1] * CANVAS_WIDTH, // z
				//top: (Y_MAX - data.cur_pos[1]) * CANVAS_HEIGHT, // z
				top: (Y_MAX - data.cur_pos[0]) * CANVAS_HEIGHT, // x
				angle: 90 + data.cur_pos[2] // y rotation
			});
		}
		if(data.aspect_ratio != null) {
			aspect_ratio = data.aspect_ratio;
			CANVAS_HEIGHT = CANVAS_WIDTH * aspect_ratio;
		}
		if(data.cur_path) {
			canvas.add(...points_to_lines(data.cur_path));
		}
		canvas.renderAll();
	}

	function create_websocket(ip_port: String) {
		let ws = new WebSocket(`ws://${ip_port}`);
		ws.onerror = (e: Event) => {
			console.log(`Websocket Failed: ${e}`);
		};
		ws.onopen = (e: Event) => {
			console.log("Succesfully Connected");
		}
		ws.onmessage = (ev: MessageEvent<any>) => {
			console.log("Received message");	
			const decoded_data: recv_data = JSON.parse(ev.data);
			console.log(decoded_data.cur_pos);
			update_canvas(decoded_data)
			console.log(indicator.left, indicator.top);
		}
		return ws;
	}

	const connect_btn = document.getElementById("ip_port_btn_id") as HTMLButtonElement;
	let ip_port_value = "localhost:8000";
	let websocket = null;
	if(connect_btn) {
		connect_btn.onclick = (e: MouseEvent) => {
			const new_ip_port_value = (document.getElementById("ip_port_id") as HTMLInputElement).value;
			if(new_ip_port_value) {
				ip_port_value = new_ip_port_value;
			}
			console.log("Connecting To: ", ip_port_value);
			websocket = create_websocket(ip_port_value);
		}
	}
}