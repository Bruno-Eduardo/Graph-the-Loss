

clc; clear all; close all;

trainLoss = csvread('data.csv');
valLoss = csvread('valLoss.csv');


figure()

xTrain = 1:length(trainLoss);
xVal = (1:length(valLoss))*length(trainLoss)/(length(valLoss));

plot(xTrain , trainLoss, xVal ,valLoss);

ylim([0,max([max(trainLoss) max(valLoss)])])
