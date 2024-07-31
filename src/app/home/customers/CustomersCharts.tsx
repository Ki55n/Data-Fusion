'use client'
import { useRef } from 'react'
import { useEffect, createRef } from 'react'
import * as d3 from 'd3'
export default function CustomersCharts() {
    return (
        <div className=" p-5 flex gap-2 ">

            <BarChart width={500} height={500}/>
            <GlobeChart width={500} height={500}/>
        </div>
    )

}

const BarChart = ({ width = 300, height = 300 }) => {
    const ref = createRef();

    useEffect(() => {
        draw();
    });

    const draw = () => {
        const data = [
            { product: 'MacBook Pro', amount: 2399 },
            { product: 'iPhone 13', amount: 799 },
            { product: 'AirPods Pro', amount: 249 },
            { product: 'Apple Watch Series 7', amount: 399 },
            { product: 'iPad Pro', amount: 1099 },
            { product: 'HomePod', amount: 299 },
        ];

        const svg = d3.select(ref.current);
        svg.selectAll('*').remove();

        const margin = { top: 20, right: 30, bottom: 40, left: 40 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const xScale = d3.scaleBand()
            .domain(data.map(d => d.product))
            .range([0, innerWidth])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.amount)])
            .nice()
            .range([innerHeight, 0]);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `$${d}`));
        g.append('g').call(d3.axisBottom(xScale))
            .attr('transform', `translate(0,${innerHeight})`)
            .selectAll("text")
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end");

        g.selectAll('rect')
            .data(data)
            .enter()
            .append('rect')
            .attr('x', d => xScale(d.product))
            .attr('y', d => yScale(d.amount))
            .attr('width', xScale.bandwidth())
            .attr('height', d => innerHeight - yScale(d.amount))
            .attr('fill', 'orange');
    };

    return (
        <div className='border-1 rounded border-primary p-3'>
            <svg width={width} height={height} ref={ref} />
        </div>
    );
};





const locations = [
    { name: 'Rio de Janeiro', coords: [-43.1729, -22.9068] },
    { name: 'Dubai', coords: [55.2708, 25.2048] },
   
    { name: 'London', coords: [-0.1276, 51.5074] },
];

const GlobeChart = ({ width = 300, height = 300 }) => {
    const ref = useRef();

    useEffect(() => {
        draw();
        return () => {
            d3.select(ref.current).selectAll('*').remove();
        };
    }, []);

    const draw = async () => {
        const svg = d3.select(ref.current)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const projection = d3.geoOrthographic()
            .scale(150)
            .center([0, 0])
            .rotate([0, -30])
            .translate([width / 2, height / 2]);

        const path = d3.geoPath().projection(projection);
        const initialScale = projection.scale();

        const globe = svg.append('circle')
            .attr('fill', '#333') // Dark color for the globe
            .attr('stroke', '#000')
            .attr('stroke-width', '0.5')
            .attr('cx', width / 2)
            .attr('cy', height / 2)
            .attr('r', initialScale);

        const data = await getWorldMap();
        const map = svg.append('g');

        map.append('g')
            .attr('class', 'countries')
            .selectAll('path')
            .data(data.features)
            .enter().append('path')
            .attr('class', d => `country_${d.properties.name.replace(' ', '_')}`)
            .attr('d', path)
            .attr('fill', 'white')
            .style('stroke', 'black')
            .style('stroke-width', 0.5)
            .style('opacity', 0.8);

        svg.call(d3.drag().on('drag', (event) => {
            const rotate = projection.rotate();
            const k = 75 / projection.scale();
            projection.rotate([
                rotate[0] + event.dx * k,
                rotate[1] - event.dy * k
            ]);
            updatePaths();
        }));

        svg.call(d3.zoom().on('zoom', (event) => {
            if (event.transform.k > 0.3) {
                projection.scale(initialScale * event.transform.k);
                updatePaths();
                globe.attr('r', projection.scale());
            } else {
                event.transform.k = 0.3;
            }
        }));

        let rotation = d3.timer((elapsed) => {
            if (!svg.select('.countries:hover').size()) {
                const rotate = projection.rotate();
                const k = 75 / projection.scale();
                projection.rotate([
                    rotate[0] - 0.1 * k,
                    rotate[1]
                ]);
                updatePaths();
            }
        });

        function updatePaths() {
            const updatedPath = d3.geoPath().projection(projection);
            svg.selectAll('path').attr('d', updatedPath);
        }
    };

    const handleLocationClick = (coords) => {
        const svg = d3.select(ref.current).select('svg');
        const projection = d3.geoOrthographic()
            .scale(150)
            .center([0, 0])
            .rotate([-coords[0], -coords[1]])
            .translate([width / 2, height / 2]);

        const path = d3.geoPath().projection(projection);
        svg.selectAll('path').attr('d', path);
    };

    return (
        <div className='border-1 rounded border-primary p-3 flex flex-col items-center'>
            <div ref={ref} />
            <div className='mt-4 flex flex-wrap justify-center'>
                {locations.map((location) => (
                    <button
                        key={location.name}
                        className='m-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700'
                        onClick={() => handleLocationClick(location.coords)}
                    >
                        {location.name}
                    </button>
                ))}
            </div>
        </div>
    );
};


const getWorldMap = async () => {
    const response = await fetch('https://static.observableusercontent.com/files/cbb0b433d100be8f4c48e19de6f0702d83c76df3def6897d7a4ccdb48d2f5f039bc3ae1141dd1005c253ca13c506f5824ae294f4549e5af914d0e3cb467bd8b0');
    return response.json();
};
