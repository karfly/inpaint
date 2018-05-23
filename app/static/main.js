function base64ToDataURI(dataURI) {
    return 'data:image/png;base64,' + dataURI
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

Api.prototype.sendImage = function (canvas_object, callback) { //$('#canvas-in')[0]
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

Api.prototype.sendMask = function (canvas_object, callback) { // $('#canvas-mask-delta')[0]
    var blob = dataURItoBlob(canvas_object.toDataURL());
    var formData = new FormData();
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


function DrawEngine(mask_selector, out_selector, width_selector) {
    this.mousePressed = false;
    this.is_allowed = false;

    this.mask_selector = mask_selector;
    this.mask_object = mask_selector[0];
    this.mask_context = mask_selector[0].getContext("2d");
    this.out_object = out_selector[0];
    this.width = width_selector.val();
}

DrawEngine.prototype.allow = function () {
    var self = this;
    if (!self.is_allowed) {
        self.mask_selector.on('mousedown touchstart', function (event) {
            console.log('start draw');
            self.startDraw(event);
        });
        self.mask_selector.on('mousemove touchmove', function (event) {
            console.log('move');
            event.preventDefault();
            self.draw(event);
        });
        self.mask_selector.on('mouseup mouseleave touchend', function () {
            console.log('end draw');
            self.applyMask();
        });
        self.is_allowed = true;
    }
};

DrawEngine.prototype.startDraw = function (event) {
    var target = event.currentTarget;
    var x_rel = (event.pageX - $(target).offset().left) / target.offsetWidth * 256;
    var y_rel = (event.pageY - $(target).offset().top) / target.offsetHeight * 256;

    this.mask_context.beginPath();
    this.mask_context.arc(x_rel, y_rel, this.width / 2, 0, 2 * Math.PI, false);
    this.mask_context.fillStyle = 'white';
    this.mask_context.fill();
    this.mask_context.closePath();

    this.mousePressed = true;
    this.lastX = x_rel;
    this.lastY = y_rel;
};

DrawEngine.prototype.draw = function (event) {
    if (this.mousePressed) {
        var target = event.currentTarget;
        var x_rel = (event.pageX - $(target).offset().left) / target.offsetWidth * 256;
        var y_rel = (event.pageY - $(target).offset().top) / target.offsetHeight * 256;

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
    }
};

DrawEngine.prototype.applyMask = function () {
    if (this.mousePressed) {
        this.mousePressed = false;
        drawHistory.push({
            'mask': this.mask_object.toDataURL()
        });
        var self = this;
        api.sendMask(this.mask_object, function (data) {
            if (drawHistory.image_id === data['image_id']) {
                var resultURI = base64ToDataURI(data['result']);
                drawHistory.update(data['step_id'], 'result', resultURI);
                if (drawHistory.cStep === data['step_id']) {
                    dataURIToImage(resultURI, function (img) {
                        fillCanvas(self.out_object, img);
                    });
                }
            }
        });
    }
};
