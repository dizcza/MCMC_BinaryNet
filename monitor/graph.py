import os
import warnings
from typing import Iterable, Tuple, Dict

import graphviz
import imageio
import numpy as np
import torch
import visdom
import math

from monitor.batch_timer import Schedule, BatchTimer


class ParameterNode(object):
    def __init__(self, source_name: str, sink_name: str, shape: torch.Size, history_heatmap=False):
        """
        :param source_name parameter source layer name
        :param sink_name: parameter sink layer name
        :param shape: parameter shape
        """
        self.source_name = source_name
        self.source_active = []
        self.sink_name = sink_name
        self.sink_active = []
        self.shape = shape
        self.idx_flipped = None
        if history_heatmap:
            self.idx_flipped = np.zeros(shape, dtype=np.float32)

    def save_activations(self, new_source: Iterable[int], new_sink: Iterable[int]):
        self.source_active = list(new_source)
        self.sink_active = list(new_sink)


class GraphMCMC(object):
    def __init__(self, named_params: Iterable[Tuple], timer: BatchTimer, history_heatmap=False):
        """
        :param named_params: named binary parameters
        :param timer: timer to schedule updates
        :param history_heatmap: draw history heatmap of all iterations?
        """
        self.param_nodes = self.parse_parameters(named_params, history_heatmap=history_heatmap)
        self.with_history_heatmap = history_heatmap
        self.timer = timer
        self.graph = graphviz.Graph(name='graph_mcmc', directory='graphs', format='png',
                                    graph_attr=dict(rankdir='LR', color='white', splines='line', nodesep='0.05'),
                                    node_attr=dict(label='', shape='circle', width='0.2'),
                                    edge_attr=dict(constraint='false'))
        try:
            self.graph.pipe(format='svg')
            self.graphviz_installed = True
        except graphviz.backend.ExecutableNotFound:
            self.graphviz_installed = False
            warnings.warn("To use graphviz features run 'sudo apt-get install graphviz'")

    @staticmethod
    def parse_parameters(named_params: Iterable[Tuple], history_heatmap: bool) -> Dict[str, ParameterNode]:
        source_name = 'input'
        pnodes = {}
        for name, param in named_params:
            pnodes[name] = ParameterNode(source_name=source_name, sink_name=name, shape=param.shape,
                                         history_heatmap=history_heatmap)
            source_name = name
        return pnodes

    def add_samples(self, param_flips: Iterable):
        for pflip in param_flips:
            assert pflip.name in self.param_nodes, f"Unexpected model parameter '{pflip.name}'. " \
                                                   f"Did you forget to pass it in the constructor?"
            pnode = self.param_nodes[pflip.name]
            if pnode.idx_flipped is not None:
                pnode.idx_flipped += pflip.get_idx_flipped().cpu().numpy()
        self.save_sample_activations(param_flips)

    @Schedule(epoch_update=0, batch_update=1)
    def save_sample_activations(self, param_flips: Iterable):
        """
        :param param_flips: Iterable of ParameterFLip
        """
        for pnode in self.param_nodes.values():
            # clear old activations
            pnode.save_activations(new_source=[], new_sink=[])
        for pflip in param_flips:
            self.param_nodes[pflip.name].save_activations(new_source=pflip.source, new_sink=pflip.sink)

    @staticmethod
    def neuron_name(layer_name: str, neuron_id: int):
        return f'{layer_name}_{neuron_id}'

    def draw_layer(self, layer_name: str, layer_size: int):
        with self.graph.subgraph(name=f'cluster_{layer_name}') as c:
            c.attr(label=layer_name)
            for neuron_id in range(layer_size):
                c.node(self.neuron_name(layer_name, neuron_id))

    def draw_model(self):
        for pnode in self.param_nodes.values():
            size_output, size_input = pnode.shape
            if pnode.source_name == "input":
                self.draw_layer(layer_name='input', layer_size=size_input)
            self.draw_layer(layer_name=pnode.sink_name, layer_size=size_output)
            self.draw_edges(source_name=pnode.source_name, source_neurons=self.central_neurons(size_input),
                            sink_name=pnode.sink_name, sink_neurons=self.central_neurons(size_output),
                            constraint='true', style='invis')
            self.draw_edges(source_name=pnode.source_name, source_neurons=pnode.source_active,
                            sink_name=pnode.sink_name, sink_neurons=pnode.sink_active,
                            penwidth='2.0')
        if self.with_history_heatmap:
            self.draw_history_heatmap()
        self.graph.node(name='epoch', label=f'MCMC draws. Epoch {self.timer.epoch}', color='white', shape='rect',
                        fontsize='20')

    def draw_history_heatmap(self, colorscheme='bugn9'):
        if len(self.param_nodes) == 0:
            return
        color_hot = int(colorscheme[-1])
        for pnode in self.param_nodes.values():
            max_flips = 1 + np.max(pnode.idx_flipped)
            size_output, size_input = pnode.shape
            for sink_id in range(size_output):
                for source_id in range(size_input):
                    flips_normed = pnode.idx_flipped[sink_id, source_id] / max_flips
                    flips_normed **= 2  # highlight peaks
                    color = math.ceil(flips_normed * color_hot)
                    width = 0.5 * flips_normed
                    self.graph.edge(self.neuron_name(layer_name=pnode.source_name, neuron_id=source_id),
                                    self.neuron_name(layer_name=pnode.sink_name, neuron_id=sink_id),
                                    colorscheme=colorscheme, color=str(color), penwidth=str(width))

    def draw_edges(self, source_name: str, source_neurons: Iterable[int], sink_name: str, sink_neurons: Iterable[int],
                   **attr):
        for input_neuron in source_neurons:
            input_neuron_name = self.neuron_name(source_name, input_neuron)
            for output_neuron in sink_neurons:
                self.graph.edge(input_neuron_name, self.neuron_name(sink_name, output_neuron), **attr)

    @staticmethod
    def central_neurons(layer_size: int):
        if layer_size % 2 == 1:
            return {layer_size // 2}
        else:
            return {layer_size // 2, layer_size // 2 - 1}

    @Schedule(epoch_update=10)
    def render(self, viz: visdom.Visdom, render_format='svg'):
        if not self.graphviz_installed or len(self.param_nodes) == 0:
            return
        self.graph.clear(keep_attrs=True)
        self.draw_model()
        if render_format == 'svg':
            svg = self.graph.pipe(format='svg').decode('utf-8')
            viz.svg(svgstr=svg, win=self.graph.name)
        elif render_format == 'png':
            filename = viz.env
            self.graph.render(filename=filename, cleanup=True)
            impath = os.path.join(self.graph.directory, filename + '.png')
            image_rendered = np.transpose(imageio.imread(impath), axes=(2, 0, 1))
            viz.image(image_rendered, win=self.graph.name)
        else:
            raise NotImplementedError()
