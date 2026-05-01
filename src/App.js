import React, { Component } from 'react';
import AppModel from './models/AppModel';
import AppPresenter from './presenters/AppPresenter';
import AppView from './views/AppView';

class App extends Component {
    constructor(props) {
        super(props);
        this.model = new AppModel();
        this.view = React.createRef();
    }

    componentDidMount() {
        // Initialize presenter with model and view
        this.presenter = new AppPresenter(this.model, this.view.current);
    }

    render() {
        return <AppView ref={this.view} />;
    }
}

export default App;
