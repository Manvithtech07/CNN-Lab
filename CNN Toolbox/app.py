import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import json

app = Flask(__name__)

global_state = {
    'model': None,
    'dataset': None,
    'class_names': None,
}

MNIST_CLASSES   = [str(i) for i in range(10)]
FASHION_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1     = nn.Linear(32 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)



def tensor_to_b64(tensor, size=None):
    t = tensor.detach().squeeze()
    if t.ndim == 3:
        t = t.mean(0)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6) * 255
    img = Image.fromarray(t.numpy().astype(np.uint8))
    if size:
        img = img.resize(size, Image.NEAREST)
    buf = BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def np_to_b64(arr_uint8, scale=1):
    img = Image.fromarray(arr_uint8)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()



def get_dataset(dataset_type, n_train=8000, n_test=500):
    from torchvision import datasets, transforms
    t  = transforms.Compose([transforms.ToTensor()])
    dd = './data'
    if dataset_type == 'mnist':
        tr = datasets.MNIST(dd, train=True,  download=True, transform=t)
        te = datasets.MNIST(dd, train=False, download=True, transform=t)
        cn = MNIST_CLASSES
    else:
        tr = datasets.FashionMNIST(dd, train=True,  download=True, transform=t)
        te = datasets.FashionMNIST(dd, train=False, download=True, transform=t)
        cn = FASHION_CLASSES
    return (torch.utils.data.Subset(tr, range(n_train)),
            torch.utils.data.Subset(te, range(n_test)), cn)


@app.route('/', methods=['GET', 'POST'])
def index():
    input_img_b64  = None
    output_img_b64 = None
    input_shape    = "N/A"
    output_shape   = "N/A"
    math_step      = None

    params = {
        'kernel_size': 3, 'stride': 1, 'padding': 0,
        'filter_type': 'random', 'activation': 'none', 'pooling': 'none',
        'dataset_source': 'upload', 'custom_weights': [0.0] * 9
    }

    if request.method == 'POST':
        try:
            params['kernel_size']    = int(request.form.get('kernel_size', 3))
            params['stride']         = int(request.form.get('stride', 1))
            params['padding']        = int(request.form.get('padding', 0))
            params['filter_type']    = request.form.get('filter_type', 'random')
            params['activation']     = request.form.get('activation', 'none')
            params['pooling']        = request.form.get('pooling', 'none')
            params['dataset_source'] = request.form.get('dataset_source', 'upload')

            custom_vals = []
            for i in range(49):
                v = request.form.get(f'w{i}')
                if v is not None:
                    custom_vals.append(float(v) if v else 0.0)
            params['custom_weights'] = custom_vals

            img      = None
            b64_data = request.form.get('image_b64_data')
            if b64_data:
                img = Image.open(BytesIO(base64.b64decode(b64_data))).convert('L')
            else:
                f = request.files.get('file')
                if f and f.filename:
                    img = Image.open(f).convert('L')

            if img:
                buf = BytesIO()
                img.save(buf, "PNG")
                input_img_b64 = base64.b64encode(buf.getvalue()).decode()

                img_t = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0)
                input_shape = list(img_t.shape)

                k    = params['kernel_size']
                conv = nn.Conv2d(1, 1, k, stride=params['stride'],
                                 padding=params['padding'], bias=False)
                with torch.no_grad():
                    wt = torch.tensor(params['custom_weights'][:k*k]).reshape(1, 1, k, k)
                    conv.weight = nn.Parameter(wt)

                try:
                    p      = params['padding']
                    padded = F.pad(img_t, (p, p, p, p))
                    patch  = padded[0, 0, 0:k, 0:k].numpy()
                    w_viz  = wt[0, 0].numpy()
                    math_step = {
                        'patch':   np.round(patch, 1).tolist(),
                        'weights': np.round(w_viz, 2).tolist(),
                        'product': np.round(patch * w_viz, 1).tolist(),
                        'sum':     round(float(np.sum(patch * w_viz)), 2)
                    }
                except Exception:
                    pass

                out = conv(img_t)
                if params['activation'] == 'relu':      out = F.relu(out)
                elif params['activation'] == 'sigmoid': out = torch.sigmoid(out)
                elif params['activation'] == 'tanh':    out = torch.tanh(out)
                if params['pooling'] == 'max':          out = F.max_pool2d(out, 2, 2)
                elif params['pooling'] == 'avg':        out = F.avg_pool2d(out, 2, 2)

                output_shape   = list(out.shape)
                output_img_b64 = tensor_to_b64(out)

        except Exception:
            pass

    return render_template('index.html',
                           input_img=input_img_b64,
                           output_img=output_img_b64,
                           params=params,
                           in_shape=input_shape,
                           out_shape=output_shape,
                           math_step=math_step)


@app.route('/api/mnist_sample')
def mnist_sample():
    dataset_type = request.args.get('dataset', 'mnist')
    class_idx    = request.args.get('class_idx', 'random')
    try:
        from torchvision import datasets, transforms
        data = (datasets.MNIST if dataset_type == 'mnist' else datasets.FashionMNIST)(
            './data', train=False, download=True, transform=transforms.ToTensor())
        cn   = MNIST_CLASSES if dataset_type == 'mnist' else FASHION_CLASSES
        pool = 1000

        if class_idx == 'random':
            idx = int(np.random.randint(0, pool))
        else:
            ci   = int(class_idx)
            idxs = [i for i in range(pool) if int(data[i][1]) == ci]
            idx  = int(np.random.choice(idxs)) if idxs else 0

        img_t, label = data[idx]
        img_np = (img_t.squeeze().numpy() * 255).astype(np.uint8)
        return jsonify({'image': np_to_b64(img_np), 'label': int(label),
                        'class_name': cn[int(label)], 'class_names': cn})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train_stream')
def train_stream():
    dataset_type = request.args.get('dataset', 'mnist')
    epochs       = int(request.args.get('epochs', 5))
    lr           = float(request.args.get('lr', 0.001))
    batch_size   = int(request.args.get('batch_size', 64))

    def generate():
        def sse(d): return f"data: {json.dumps(d)}\n\n"
        try:
            yield sse({'type': 'status', 'message': 'Loading dataset...'})
            train_d, test_d, cn = get_dataset(dataset_type)
            yield sse({'type': 'status',
                       'message': f'Loaded {len(train_d)} train / {len(test_d)} val samples.'})

            train_loader = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True)
            test_loader  = torch.utils.data.DataLoader(test_d,  batch_size=batch_size)

            model = SimpleCNN()
            opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            crit  = nn.CrossEntropyLoss()
            hist  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for ep in range(epochs):
                model.train()
                tl = tc = tt = 0
                for X, y in train_loader:
                    opt.zero_grad()
                    out  = model(X)
                    loss = crit(out, y)
                    loss.backward()
                    opt.step()
                    tl += loss.item()
                    tc += (out.argmax(1) == y).sum().item()
                    tt += len(y)

                model.eval()
                vl = vc = vt = 0
                with torch.no_grad():
                    for X, y in test_loader:
                        out  = model(X)
                        loss = crit(out, y)
                        vl += loss.item()
                        vc += (out.argmax(1) == y).sum().item()
                        vt += len(y)

                tr_loss = round(tl / len(train_loader), 4)
                tr_acc  = round(tc / tt * 100, 2)
                vl_loss = round(vl / len(test_loader), 4)
                vl_acc  = round(vc / vt * 100, 2)
                hist['train_loss'].append(tr_loss)
                hist['val_loss'].append(vl_loss)
                hist['train_acc'].append(tr_acc)
                hist['val_acc'].append(vl_acc)

                yield sse({'type': 'epoch', 'epoch': ep + 1, 'total': epochs,
                           'train_loss': tr_loss, 'val_loss': vl_loss,
                           'train_acc': tr_acc,   'val_acc': vl_acc,
                           'history': hist})

            global_state['model']       = model
            global_state['dataset']     = dataset_type
            global_state['class_names'] = cn

            model.eval()
            s_img, s_lbl = test_d[0]
            inp = s_img.unsqueeze(0)
            with torch.no_grad():
                out = model(inp)

            probs = F.softmax(out, 1)[0].numpy().tolist()
            pred  = int(out.argmax(1))
            in_b64 = tensor_to_b64(s_img, (112, 112))

            yield sse({'type': 'complete',
                       'probs': probs, 'pred_class': pred,
                       'true_label': int(s_lbl), 'class_names': cn,
                       'input_img': in_b64, 'history': hist,
                       'dataset': dataset_type})

        except Exception as e:
            yield sse({'type': 'error', 'message': str(e)})

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    if global_state['model'] is None:
        return jsonify({'error': 'No model trained. Train a model first.'}), 400
    data    = request.json
    img_b64 = data.get('image')
    try:
        img    = Image.open(BytesIO(base64.b64decode(img_b64))).convert('L').resize((28, 28))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t  = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        model  = global_state['model']
        model.eval()
        with torch.no_grad():
            out   = model(img_t)
            probs = F.softmax(out, 1)[0].numpy().tolist()
            pred  = int(out.argmax(1))
        cn      = global_state['class_names']
        orig_u8 = (img_np * 255).astype(np.uint8)
        return jsonify({
            'probs': probs, 'pred_class': pred,
            'pred_name': cn[pred], 'class_names': cn,
            'image': np_to_b64(orig_u8, scale=4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True, threaded=True)