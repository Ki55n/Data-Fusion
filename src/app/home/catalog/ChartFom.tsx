'use client'
import React from "react";
import {Button} from "@nextui-org/react";
import {Divider} from "@nextui-org/divider";
import { IoBarChartOutline } from "react-icons/io5";
import { FaChartLine } from "react-icons/fa6";
import { FaChartPie } from "react-icons/fa6";
import {AiFillDatabase} from "react-icons/ai";
import { CiUser } from "react-icons/ci";
import { LuPackage } from "react-icons/lu";
// d3 imports for pie chart, line chart and bar chart
import {select} from 'd3-selection';
import {arc} from 'd3-shape';
import {scaleOrdinal} from 'd3-scale';
import {schemeCategory10} from 'd3-scale-chromatic';
import {pie} from 'd3-shape';
import {line} from 'd3-shape';
import {scaleLinear} from 'd3-scale';
import {axisBottom} from 'd3-axis';
import {axisLeft} from 'd3-axis';
import {scaleBand} from 'd3-scale';
import {max} from 'd3-array';
import {select as d3Select} from 'd3-selection';
import {transition} from 'd3-transition';
import {easeLinear} from 'd3-ease';

export const ChartFom = ({onSave}) => {
    const [selectedDataSet, setSelectedDataSet] = React.useState<string>();
    const [selectedChartType, setSelectedChartType] = React.useState<string>();
    const [description, setDescription] = React.useState<string>();
    const chartButtons = [
        {
            name: 'Pie Chart',
            icon: <FaChartPie/>,
            value: 'pie'
        },
        {
            name: 'Line Chart',
            icon: <FaChartLine/>,
            value: 'line'
        },
        {
            name: 'Bar Chart',
            icon: <IoBarChartOutline/>,
            value: 'bar'

        }]
    const dataSources = [
        {
            name: 'Catalog',
            value: 'catalog',
            icon: <AiFillDatabase/>
        },
        {
            name: 'Orders',
            value: 'orders',
            icon: <LuPackage/>
        },
        {
            name: 'Customers',
            value: 'customers',
            icon: <CiUser/>
        }
    ]
    const onGenerateChart = (e: React.FormEvent) => {
        e.preventDefault();
        onSave({
            selectedDataSet,
            selectedChartType,
            description
        })
    };

    return <form className={'flex flex-col gap-4'} >
        <label className="">Datasets to use </label>
        <div className={'flex gap-3 justify-center'}>
            {dataSources.map((button, index) => (
                <Button
                    key={index}
                    className="flex gap-1 flex-col h-24"
                    onClick={() => setSelectedDataSet(button.value)}
                    color={selectedDataSet === button.value ? 'primary' : 'default'}
                >
                    {button.icon}
                    {button.name}
                </Button>
            ))}
        </div>
        <label className="">Type of chart</label>
        <div className={'flex gap-3 justify-center'}>
            {chartButtons.map((button, index) => (
                <Button
                    key={index}
                    className="flex gap-1 flex-col h-24"
                    onClick={() => setSelectedChartType(button.value)}
                    color={selectedChartType === button.value ? 'primary' : 'default'}
                >
                    {button.icon}
                    {button.name}
                </Button>
            ))}
        </div>

        <Divider/>
        <textarea
            required
            className="form-input mt-1 block w-full"
            name={'description'}
            placeholder={'description'}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
        />
        <Button
            type="submit"
            className=" w-full"
            color={'primary'}
            isDisabled={
                !selectedDataSet || !selectedChartType || !description
            }
            onClick={onGenerateChart}
        >
            Generate Chart
        </Button>
    </form>
}

