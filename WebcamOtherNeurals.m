clear all
close all
clc
camera = webcam(1);
%camera.Resolution='640x480'
% check different neural networks at https://la.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
%net = googlenet;
net = nasnetlarge

sizeLayers = size(net.Layers);
classes= net.Layers(sizeLayers(1),sizeLayers(2)).Classes;

inputSize = net.Layers(1).InputSize(1:2);


h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
ax2.PositionConstraint = 'innerposition';

while ishandle(h)
    % Display and classify the image
    im = snapshot(camera);
    image(ax1,im)
    im = imresize(im,inputSize);
    [label,score] = classify(net,im);
    title(ax1,{char(label),num2str(max(score),2)});

    % Select the top five predictions
    [~,idx] = sort(score,'descend');
    idx = idx(5:-1:1);
    scoreTop = score(idx);
    classNamesTop = string(classes(idx));

    % Plot the histogram
    barh(ax2,scoreTop)
    title(ax2,'Top 5')
    xlabel(ax2,'Probability')
    xlim(ax2,[0 1])
    yticklabels(ax2,classNamesTop)
    ax2.YAxisLocation = 'right';

    drawnow
end