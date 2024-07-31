'use client'
import { useEffect, createRef } from 'react'
import * as d3 from 'd3'
import { FaRegLightbulb } from "react-icons/fa";
import { Image } from "@nextui-org/image";
import { FaVolumeUp } from "react-icons/fa";


export default function CatalogCharts() {
    
    return (
        <div className="grid gap:  grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            <PopularProduct data={ {
        id: '1',
        name: 'Leather Gloves',
        description: 'Premium quality leather gloves for winter, providing both warmth and style.',
        image: '/gloves.jpeg'
    }} />
            <DistributionHistogram />
            <PieChart />
            <LineChart />
            <BarChart />
        </div>
    )
}

const LineChart = ({ width = 300, height = 300 }) => {
    const ref = createRef();

    useEffect(() => {
        draw();
    }, []);

    const draw = () => {
        const data = [12, 5, 6, 6, 9, 10];
        const svg = d3.select(ref.current);
        svg.selectAll('*').remove();

        const margin = { top: 20, right: 20, bottom: 50, left: 60 },
            chartWidth = width - margin.left - margin.right,
            chartHeight = height - margin.top - margin.bottom;

        const xScale = d3.scaleLinear()
            .domain([0, data.length - 1])
            .range([0, chartWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data)])
            .nice()
            .range([chartHeight, 0]);

        const line = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d));

        const g = svg.append('g')
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Line path
        g.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', 'teal')
            .attr('stroke-width', 2)
            .attr('d', line);

        // X Axis
        g.append('g')
            .attr('transform', `translate(0,${chartHeight})`)
            .call(d3.axisBottom(xScale).ticks(data.length))
            .selectAll('text')
            .attr('class', 'text-xs text-gray-600')
            .style('text-anchor', 'middle');

        // Y Axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .selectAll('text')
            .attr('class', 'text-xs text-gray-600');

        // X Axis Label
        svg.append('text')
            .attr('transform', `translate(${width / 2}, ${height - 10})`)
            .style('text-anchor', 'middle')
            .attr('class', 'text-sm font-semibold text-gray-700')
            .text('Data Index');

        // Y Axis Label
        svg.append('text')
            .attr('transform', `translate(${margin.left - 40}, ${height / 2}) rotate(-90)`)
            .style('text-anchor', 'middle')
            .attr('class', 'text-sm font-semibold text-gray-700')
            .text('Value');
    };

    return (
        <div className="border border-slate-700 rounded p-3  shadow-md">
            <h3 className="text-lg font-bold  mb-2">Data Trend Line Chart</h3>
            <p className="text-sm text-slate-400 mb-3">
                This line chart visualizes the trend of data points over a series of indices. Use this chart to observe changes and patterns in data values.
            </p>
            <svg ref={ref} width={width} height={height} className="w-full h-64" />
        </div>
    );
};
const PieChart = ({ width = 200, height = 200 }) => {
    const ref = createRef();

    useEffect(() => {
        draw();
    }, []);

    const draw = () => {
        const data = [12, 5, 6, 6, 9, 10];
        const svg = d3.select(ref.current);
        svg.selectAll('*').remove();

        const radius = Math.min(width, height) / 2;
        const color = d3.scaleOrdinal(['#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#e41a1c']);
        const pie = d3.pie();
        const arc = d3.arc().innerRadius(0).outerRadius(radius);

        const g = svg.append("g")
            .attr("transform", `translate(${width / 2},${height / 2})`);

        const arcs = g.selectAll(".arc")
            .data(pie(data))
            .enter()
            .append("g")
            .attr("class", "arc");

        arcs.append("path")
            .attr("d", arc)
            .attr("fill", (d, i) => color(i));

        // Labels
        arcs.append("text")
            .attr("transform", d => `translate(${arc.centroid(d)})`)
            .attr("dy", ".35em")
            .attr("text-anchor", "middle")
            .attr("class", "text-xs font-medium text-white")
            .text(d => d.data);

        // Legend
        const legend = g.append("g")
            .attr("transform", `translate(${radius + 20}, -${height / 2})`);

        data.forEach((value, i) => {
            legend.append("rect")
                .attr("x", 0)
                .attr("y", i * 20)
                .attr("width", 15)
                .attr("height", 15)
                .attr("fill", color(i));

            legend.append("text")
                .attr("x", 20)
                .attr("y", i * 20 + 12)
                .attr("class", "text-xs text-gray-700")
                .text(`Category ${i + 1}: ${value}`);
        });
    };

    return (
        <div className="border border-slate-700 rounded p-3 shadow-md">
            <h3 className="text-lg font-bold  mb-2">Product Category Distribution</h3>
            <p className="text-sm text-slate-400 mb-3">
                This pie chart displays the distribution of product categories. Each slice represents a category's proportion relative to the total.
            </p>
            <svg ref={ref} width={width} height={height} className="w-full h-64" />
        </div>
    );
};

const BarChart = ({ width = 300, height = 300 }) => {
    const ref = createRef();

    useEffect(() => {
        draw();
    }, []);

    const draw = () => {
        const data = [12, 5, 6, 6, 9, 10];
        const svg = d3.select(ref.current);
        svg.selectAll('*').remove();

        const margin = { top: 20, right: 20, bottom: 50, left: 60 },
            chartWidth = width - margin.left - margin.right,
            chartHeight = height - margin.top - margin.bottom;

        const xScale = d3.scaleBand()
            .domain(d3.range(data.length))
            .range([0, chartWidth])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data)])
            .nice()
            .range([chartHeight, 0]);

        const g = svg.append('g')
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Bars
        g.selectAll('rect')
            .data(data)
            .enter()
            .append('rect')
            .attr('x', (d, i) => xScale(i))
            .attr('y', d => yScale(d))
            .attr('width', xScale.bandwidth())
            .attr('height', d => chartHeight - yScale(d))
            .attr('fill', 'teal');

        // X Axis
        g.append('g')
            .attr('transform', `translate(0,${chartHeight})`)
            .call(d3.axisBottom(xScale).tickFormat(i => `Item ${i + 1}`))
            .selectAll('text')
            .attr('class', 'text-xs text-gray-600')
            .style('text-anchor', 'middle');

        // Y Axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .selectAll('text')
            .attr('class', 'text-xs text-gray-600');

        // X Axis Label
        svg.append('text')
            .attr('transform', `translate(${width / 2}, ${height - 5})`)
            .style('text-anchor', 'middle')
            .attr('class', 'text-sm font-semibold text-gray-700')
            .text('Item');

        // Y Axis Label
        svg.append('text')
            .attr('transform', `translate(${margin.left - 40}, ${height / 2}) rotate(-90)`)
            .style('text-anchor', 'middle')
            .attr('class', 'text-sm font-semibold text-gray-700')
            .text('Value');
    };

    return (
        <div className="border border-slate-700 rounded p-3 shadow-md">
            <h3 className="text-lg font-bold  mb-2">Value Distribution Bar Chart</h3>
            <p className="text-sm text-slate-400 mb-3">
                This bar chart displays the distribution of values across different items. Use this visualization to compare the magnitude of different data points.
            </p>
            <svg ref={ref} width={width} height={height} className="w-full h-64" />
        </div>
    );
};
const PopularProduct = ({ data }) => {
    return <div className="border border-slate-700 p-3 rounded">
        <h3 className="flex  font-bold justify-between">Your most Popular Item <FaVolumeUp /></h3>
        <p className="text-sm text-slate-400">Discover your best-selling product. Identify the top-performing item in your inventory helping you focus on what customers love most.</p>
        <div className="flex flex-col items-center justify-center my-3">
            <Image src={data.image} width={100} height={100} />
            <div>
                <h5 className="text-center">{data.name}</h5>
                <p className="text-center">{data.description}</p>
            </div>
            <br />
            <div className="flex items-center justify-center bg-blue-400 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded shadow-md transition duration-300">
      <FaRegLightbulb className="text-yellow-300 mr-2" />
      <p>Prompt: Why is it so popular?</p>
    </div>
        </div>
    </div>
}

const DistributionHistogram = ({ width = 300, height = 300 }) => {
    const ref = createRef();

    useEffect(() => {
        draw();
    }, []);

    const draw = () => {
        const data = [12, 5, 6, 6, 9, 10];
        const svg = d3.select(ref.current);
        svg.selectAll('*').remove();

        const margin = { top: 20, right: 20, bottom: 50, left: 60 },
            chartWidth = width - margin.left - margin.right,
            chartHeight = height - margin.top - margin.bottom;

        const xScale = d3.scaleBand()
            .domain(d3.range(data.length))
            .range([0, chartWidth])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data)])
            .nice()
            .range([chartHeight, 0]);

        const g = svg.append('g')
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Bars
        g.selectAll('rect')
            .data(data)
            .enter()
            .append('rect')
            .attr('x', (d, i) => xScale(i))
            .attr('y', d => yScale(d))
            .attr('width', xScale.bandwidth())
            .attr('height', d => chartHeight - yScale(d))
            .attr('fill', 'orange');

        // X Axis
        g.append('g')
            .attr('transform', `translate(0,${chartHeight})`)
            .call(d3.axisBottom(xScale).tickFormat(i => `Product ${i + 1}`))
            .selectAll('text')
            .attr('class', 'text-xs text-gray-600')
            .style('text-anchor', 'middle');

        // Y Axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .selectAll('text')
            .attr('class', 'text-xs text-gray-600');

        // X Axis Label
        svg.append('text')
            .attr('transform', `translate(${width / 2}, ${height - 5})`)
            .style('text-anchor', 'middle')
            .attr('class', 'text-sm font-semibold text-gray-700')
            .text('Product');

        // Y Axis Label
        svg.append('text')
            .attr('transform', `translate(${margin.left - 40}, ${height / 2}) rotate(-90)`)
            .style('text-anchor', 'middle')
            .attr('class', 'text-sm font-semibold text-gray-700')
            .text('Price ($)');
    };

    return (
        <div className="border border-slate-700 rounded p-3  shadow-md">
            <h3 className="flex justify-between font-bold text-lg ">
                Price Distribution Histogram <FaVolumeUp />
            </h3>
            <p className="text-sm text-slate-400 mt-2">
                Closely examine the price distribution of your products. This chart offers a visual breakdown of product prices, helping you identify pricing trends, popular price ranges, and potential pricing strategies.
            </p>
            <svg ref={ref} width={width} height={height} className="w-full h-64 mt-3" />
        </div>
    );
};

