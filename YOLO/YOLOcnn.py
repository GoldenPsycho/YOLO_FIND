# app for find objects on image

# names of dataset classes -- https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names

# config for tiny-variation of YOLO network -- https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# weights for tiny-variation of YOLO network -- https://raw.githubusercontent.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights



import cv2

import numpy as np

def applay_yolo_object_detection(image_to_process):
	'''
	search and detection of object's coordinates on img
	:param image_to_process: primary img
	:return: image with marked objects and captions for them
	'''
	height, width, depth = image_to_process.shape

	blob = cv2.dnn.blobFromImage(image_to_process,1/255,(608,608),(0,0,0),swap=True,crop=False)

	net.setInput(blob)

	outs = net.forward(out_layers)

	class_indexes, class_scores, boxes = ([] for i in range(3))

	objects_count = 0

#start search of objects on image

for out in outs:
	for obj in out:

		scores = obj[5:]

		class_index = np.argmax(scores)

		class_score = scores[class_index]

		if class_score > 0:

			center_x = int(obj[0]*width)
			center_y = int(obj[1]*height)

			obj_width = int(obj[2]*width)
			obj_height = int(obj[3]*height)

			box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
			boxes.append(box)

			class_indexes.append(class_index)
			class_scores.append(float(class_score))

#Selection

chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_score, 0.0, 0.4)

for box_index in chosen_boxes:

	box_index = box_index[0]
	box = boxes[box_index]

	class_index = class_indexes[box_index]

# For debuging drawings, objects that are assigned to the required classes

	if classes[class_index] in classes_to_look_for:

		objects_count += 1

		image_to_process = draw_object_bounding_box(image_to_process, class_index, box)


final_image = draw_object_count(image_to_process, objects_count)

# return final_image


def draw_object_bounding_box(image_to_process, index, box):
	'''
	drawing borders of objects with signature
	:param image_to_process: primary img
	:param index: index of obj by YOLO class
	:param box: coordinates of area around obj
	:return: image with markerd objects
	'''
	x, y, w ,h = box
	start = (x, y)
	end = (x + w, y + h)
	color = (0, 255, 0)
	width = 2
	final_image = cv2.rectangle(image_to_process, start, end, color, width)


	start = (x, y - 10)
	font_size = 1
	font = cv2.FONT_HERSHEY_SIMPLEX
	width = 2
	text = classes[index]
	final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

	return final_image


def draw_object_count(image_to_process, objects_count):
	start = (45, 150)
	font_size = 1.5
	font = cv2.FONT_HERSHEY_SIMPLEX
	width = 3
	text = "Objects found: " + str(objects_count)

	# print text with border 

	white_color = (255, 255, 255)
	black_outline_color = (0, 0, 0)
	final_image = cv2.putText(image_to_process, text, start, font, font_size, black_outline_color, width * 3, cv2.LINE_AA)
	final_image = cv2.putText(final_image, text, start, font, font_size, white_color, width, cv2.LINE_AA)

	return final_image

def start_image_object_detection():
	'''
	analitica of image

	'''
	try:
		image = cv2.imread("assets/truck_captcha.png")
		image = applay_yolo_object_detection(image)
# image to screen
		cv2.imshow("Image", image)
		if cv2.waitKey(0):
			cv2.destroyAllWindows()
	except KeyboardInterrupt:
		pass

if __name__ == '__main__':
	#download weights yolo from files of network config
	net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg","yolov4-tiny.weights")
	layer_names = net.getLayerNames()
	out_layers_indexes = net.getUnconnectedOutLayers()
	out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]
	#download classes objects 
	with open("coco.names.text") as file:
		classes = file.read().split("\n")

	classes_to_look_for = ["truck","person"]
	start_image_object_detection()



	def start_video_object_detection():
		'''
		catching and analitica in realtime
		'''

		while True:
			try:
				video_camera_capture = cv2.VideoCapture("http://81.130.136.82:82/mjpg/video.mjpg")

				while video_camera_capture.isOpened():
					ret, frame = video_camera_capture.read()
					if not ret:
						break
				video_camera_capture.release()
				cv2.destroyAllWindows()
			except KeyboardInterrupt:
				pass
















