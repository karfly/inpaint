function base64ToDataURI(base64) {
    return 'data:image/png;base64,' + base64
}

function blobToDataURI(blob, callback) {
    var reader = new FileReader();
    reader.onload = function (e) {
        var dataUrl = e.target.result;
        callback(dataUrl);
    };
    reader.readAsDataURL(blob);
}

function dataURItoBlob(dataURI) {
    var binary = atob(dataURI.split(',')[1]);
    var array = [];
    for (var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    return new Blob([new Uint8Array(array)], {type: 'image/png'});
}

function dataURIToImage(dataUrl, callback) {
    var img = new Image();
    img.onload = function (e) {
        var image = e.target;
        callback(image);
    };
    img.src = dataUrl;
}

function blobToImage(blob, callback) {
    blobToDataURI(blob, function (dataUrl) {
        dataURIToImage(dataUrl, callback);
    });
}


function History() {
    this.cPushArray = [];
    this.cStep = -1;
}

History.prototype.push = function (data) {
    this.cStep++;
    if (this.cStep < this.cPushArray.length) {
        this.cPushArray.length = this.cStep;
    }
    this.cPushArray.push(data);
};

History.prototype.update = function (step_id, key, value) {
    this.cPushArray[step_id][key] = value
};


History.prototype.undo = function () {
    if (this.cStep > 0) {
        this.cStep--;
    }
    return this.cPushArray[this.cStep];
};


History.prototype.redo = function () {
    if (this.cStep < this.cPushArray.length - 1) {
        this.cStep++;
    }
    return this.cPushArray[this.cStep];
};

History.prototype.clear = function () {
    this.cPushArray = [];
    this.cStep = -1;
};


function Api() {

}

Api.prototype.sendImage = function (canvas_object, callback) {
    var blob = dataURItoBlob(canvas_object.toDataURL());
    var formData = new FormData();
    formData.append('image', blob, 'image.png');

    $.ajax({
        url: '/add_image',
        type: 'POST',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        success: callback
    });
};

Api.prototype.sendMask = function (image_id, canvas_object, callback) {
    var blob = dataURItoBlob(canvas_object.toDataURL());
    var formData = new FormData();
    formData.append('image_id', image_id);
    formData.append('step_id', drawHistory.cStep);
    formData.append('mask', blob, 'mask.png');

    $.ajax({
        url: '/apply_mask',
        type: 'POST',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        success: callback
    });
};


function DrawEngine(mask_object, out_object, width_object) {
    this.mousePressed = false;
    this.is_allowed = false;

    this.mask_selector = $(mask_object);
    this.mask_object = mask_object;
    this.mask_context = mask_object.getContext("2d");
    this.out_object = out_object;
    this.width = $(width_object).val();
}

DrawEngine.prototype.allow = function () {
    var self = this;
    if (!self.is_allowed) {
        self.mask_selector.on('mousedown touchstart', function (event) {
            // console.log('start draw');
            self.mousePressed = true;
            var rel_coords = self.relativeCoords(event);
            self.startDraw(rel_coords[0], rel_coords[1]);
        });
        self.mask_selector.on('mousemove touchmove', function (event) {
            if (self.mousePressed) {
                // console.log('draw move');
                var rel_coords = self.relativeCoords(event);
                self.draw(rel_coords[0], rel_coords[1]);
            }
            event.preventDefault();
        });
        self.mask_selector.on('mouseup mouseleave touchend', function () {
            // console.log('end draw');
            if (self.mousePressed) {
                self.mousePressed = false;
                self.applyMask();
            }
        });
        self.is_allowed = true;
    }
};

DrawEngine.prototype.relativeCoords = function (event) {
    var target = event.currentTarget;
    var pageX, pageY;
    if (event.targetTouches) { // this is touch
        if (event.targetTouches.length === 1) { // not gestures
            pageX = event.targetTouches[0].pageX;
            pageY = event.targetTouches[0].pageY;
        }
    } else { // this is desktop
        pageX = event.pageX;
        pageY = event.pageY;
    }
    var x_rel = (pageX - $(target).offset().left) / target.offsetWidth * 256;
    var y_rel = (pageY - $(target).offset().top) / target.offsetHeight * 256;
    // console.log(x_rel, y_rel);
    return [x_rel, y_rel];
};

DrawEngine.prototype.startDraw = function (x_rel, y_rel) {
    this.mask_context.beginPath();
    this.mask_context.arc(x_rel, y_rel, this.width / 2, 0, 2 * Math.PI, false);
    this.mask_context.fillStyle = 'white';
    this.mask_context.fill();
    this.mask_context.closePath();

    this.lastX = x_rel;
    this.lastY = y_rel;
};

DrawEngine.prototype.draw = function (x_rel, y_rel) {
    this.mask_context.beginPath();
    this.mask_context.strokeStyle = 'white';
    this.mask_context.lineWidth = this.width;
    this.mask_context.lineJoin = "round";
    this.mask_context.moveTo(this.lastX, this.lastY);
    this.mask_context.lineTo(x_rel, y_rel);
    this.mask_context.closePath();
    this.mask_context.stroke();

    this.lastX = x_rel;
    this.lastY = y_rel;
};

DrawEngine.prototype.applyMask = function () {
    drawHistory.push({
        'mask': this.mask_object.toDataURL()
    });
    var self = this;
    api.sendMask(drawHistory.image_id, this.mask_object, function (data) {
        if (drawHistory.image_id === data['image_id']) {
            var resultURI = data['result'];
            drawHistory.update(data['step_id'], 'result', resultURI);
            if (drawHistory.cStep === data['step_id']) {
                dataURIToImage(resultURI, function (img) {
                    fillCanvas(self.out_object, img);
                });
            }
        }
    });
};
